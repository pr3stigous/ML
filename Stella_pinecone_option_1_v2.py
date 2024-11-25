"""
v2
-this code asked the user to specify the rows from the sqlite DB to index into pinecone
-it returns the relevant text chunks and associated reference

####
random note after v2 changes:
-maybe i should already have an index initialized with all the processed data
completely so i dont have to do this again; so then maybe i don't need to select 
which rows i want initialized in pinecone if i have it done already
-maybe increase the chunk size? 
####

changes to make for v3:
-integrate a small LLM?

changes to make for v4:
-multiple indexes

version idk:
-i would like to compare RAPTOR with sentence window retrieval

changes to make for v5:
-integrate 
-metadata filtering
-query per index

changes to make for v6:
-testing out multiple embedding models
"""

"""
google colab install

!pip install transformers
!pip install torch
!pip install scikit-learn
!pip install beautifulsoup4
!pip install pinecone-client
!pip install bs4 lxml
"""

import os
import sqlite3
from getpass import getpass
from bs4 import BeautifulSoup
import torch
from transformers import AutoModel, AutoTokenizer
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Initialize model and tokenizer globally
print("Initializing model and tokenizer...")
MODEL = AutoModel.from_pretrained(MODEL_DIR).cuda().eval()
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
print("Model and tokenizer initialized.")

def get_row_ids():
    global SELECTED_ROW_IDS
    if SELECTED_ROW_IDS is not None:
        return SELECTED_ROW_IDS
    
    choice = input("Enter 'M' for multiple specific rows or 'R' for a range of rows: ").upper()
    if choice == 'M':
        return [int(x) for x in input("Enter row IDs separated by commas: ").split(',')]
    elif choice == 'R':
        start = int(input("Enter start row ID: "))
        end = int(input("Enter end row ID: "))
        return list(range(start, end + 1))
    else:
        print("Invalid choice. Using default row IDs.")
        return [1, 2, 3, 4, 5, 7]

def clean_html(conn, row_id):
    with conn:
        c = conn.cursor()
        c.execute(f"SELECT {HTML_COLUMN_NAME}, {HASH_COLUMN_NAME} FROM {TABLE_NAME} WHERE rowid = ?", (row_id,))
        
        row = c.fetchone()
        soup = BeautifulSoup(row[0], 'lxml')
        for tag in soup.find_all(['nav', 'aside', 'footer', 'header', 'a']):
            tag.decompose()
        
        texts = soup.stripped_strings
        cleaned_text = '\n'.join(texts)
        return cleaned_text, row[1]

def embed_text(ids, texts, is_query=False):
    vectors = []
    new_ids = []
    chunks_dict = {}
    try:
        for id, text in zip(ids, texts):
            chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE-CHUNK_OVERLAP)]
            for i, chunk in enumerate(chunks):
                prompt = "Instruct: Retrieve semantically similar text.\nQuery: " + chunk if is_query else chunk
                with torch.no_grad():
                    input_data = TOKENIZER(prompt, truncation=True, padding="longest", max_length=512, return_tensors='pt')
                    input_data = {k: v.cuda() for k, v in input_data.items()}
                    attention_mask = input_data['attention_mask']
                    last_hidden_state = MODEL(**input_data)[0]
                    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                    vector = (last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]).cpu().numpy()
                vectors.append(vector.flatten().tolist())
                chunk_id = f'{id}_chunk_{i}'
                new_ids.append(chunk_id)
                chunks_dict[chunk_id] = chunk
        return new_ids, vectors, chunks_dict
    except Exception as e:
        print(f"Error in embedding text: {str(e)}")
        return new_ids, vectors, chunks_dict

def create_pinecone_index(pc, index_name, dimension):
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")
    except Exception as e:
        print(f"Error creating Pinecone index: {str(e)}")

def upsert_to_pinecone(pc, index_name, ids, vectors):
    try:
        index = pc.Index(index_name)
        to_upsert = list(zip(ids, vectors))
        index.upsert(vectors=to_upsert)
        print(f"Vectors upserted to index '{index_name}' successfully.")
    except Exception as e:
        print(f"Error upserting to Pinecone: {str(e)}")
        print(f"First vector shape: {len(vectors[0])}")

def query_pinecone(pc, index_name, query_vector, top_k=5):
    try:
        index = pc.Index(index_name)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {str(e)}")
        return None

def get_url_from_hash(conn, hash_id):
    with conn:
        c = conn.cursor()
        c.execute(f"SELECT {URL_COLUMN_NAME} FROM {TABLE_NAME} WHERE {HASH_COLUMN_NAME} = ?", (hash_id,))
        result = c.fetchone()
        return result[0] if result else None

def process_documents(row_ids, pc, index_name):
    texts = []
    ids = []
    with sqlite3.connect(DB_NAME) as conn:
        for row_id in row_ids:
            cleaned_text, hash_id = clean_html(conn, row_id)
            texts.append(cleaned_text)
            ids.append(hash_id)

    doc_ids, doc_vectors, chunks_dict = embed_text(ids, texts)
    print(f"Vector dimension: {len(doc_vectors[0])}")
    upsert_to_pinecone(pc, index_name, doc_ids, doc_vectors)
    return chunks_dict

def perform_query(query, pc, index_name, chunks_dict):
    _, query_vector, _ = embed_text([1], [query], is_query=True)
    results = query_pinecone(pc, index_name, query_vector[0])

    if results:
        print(f'Results for query: "{query}"')
        with sqlite3.connect(DB_NAME) as conn:
            for match in results['matches']:
                chunk_id = match["id"]
                original_id = chunk_id.rsplit('_chunk_', 1)[0]
                url = get_url_from_hash(conn, original_id)
                chunk_text = chunks_dict.get(chunk_id, "Chunk text not found")
                print(f'ID: {chunk_id}, Score: {match["score"]}')
                print(f'URL: {url}')
                print(f'Chunk: {chunk_text}\n')

def main():
    api_key = getpass('Enter your Pinecone API key: ')
    pc = Pinecone(api_key=api_key)
    index_name = INDEX_NAME
    dimension = 1536

    create_pinecone_index(pc, index_name, dimension)

    row_ids = get_row_ids()
    chunks_dict = process_documents(row_ids, pc, index_name)

    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        perform_query(query, pc, index_name, chunks_dict)


if __name__ == "__main__":
    main()

