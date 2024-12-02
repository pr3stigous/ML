# -*- coding: utf-8 -*-
"""
this code is to query a pinecone database in google colab
it integrates with a sqlite database during retrieval process to provide additional metadata (weblink)
this code uses stella as an embedding model
v2 changes:
-also print out the retrieved vector in natural language,
not just the numerical representation
v3 changes:
-v2 version did not have a query prompt built in
v4 changes:
-used sonnets recommendation after feeding it embedding documentation
v5 changes:
-
"""

!pip install transformers
!pip install torch
!pip install pinecone-client
!pip install scikit-learn

import sqlite3
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from getpass import getpass
from pinecone import Pinecone
from sklearn.preprocessing import normalize

# Initialize SQLite connection
db_name = 'data2.sqlite'  # Replace with your actual database name if different
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Initialize Pinecone
api_key = getpass("Enter your Pinecone API key: ")
pc = Pinecone(api_key=api_key)

# Connect to the Pinecone index
index_name = "tester3"
index = pc.Index(index_name)

# Initialize model and tokenizer
MODEL_DIR = 'dunzhang/stella_en_1.5B_v5'
print("Initializing model and tokenizer...")
MODEL = AutoModel.from_pretrained(MODEL_DIR).cuda().eval()
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
print("Model and tokenizer initialized.")

# Add the linear layer to the model
vector_dim = 1536
MODEL.vector_linear = torch.nn.Linear(in_features=MODEL.config.hidden_size, out_features=vector_dim).cuda()

def get_embedding(text, is_query=False):
    """
    Generate embedding for the given text using the initialized model.
    If is_query is True, prepend the query prompt.
    """
    query_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "
    if is_query:
        text = query_prompt + text

    inputs = TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        attention_mask = inputs["attention_mask"]
        last_hidden_state = MODEL(**inputs)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embeddings = normalize(MODEL.vector_linear(embeddings).cpu().numpy())
    return embeddings[0].tolist()  # Convert numpy array to list

def decode_vector(vector):
    """
    Decode a vector back into text using the model's tokenizer.
    """
    # Convert the vector to a tensor and reshape it
    tensor = torch.tensor(vector).unsqueeze(0).to('cuda')

    # Use the model to generate text from the vector
    with torch.no_grad():
        outputs = MODEL.generate(inputs_embeds=tensor, max_length=100)

    # Decode the generated tokens back to text
    decoded_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return decoded_text

def query_pinecone(query_vector, top_k=5):
    """
    Query the Pinecone index and return the most relevant results.
    """
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return results

def get_url_from_sqlite(hash_value):
    """
    Query the SQLite database to get the URL associated with a hash value.
    """
    cursor.execute("SELECT url FROM webdata WHERE hash = ?", (hash_value,))
    result = cursor.fetchone()
    return result[0] if result else None

def process_query(query_text, top_k=5):
    query_vector = get_embedding(query_text, is_query=True)
    results = query_pinecone(query_vector, top_k)

    processed_results = []
    for match in results['matches']:
        id_parts = match['id'].split('_')
        if len(id_parts) > 1:
            hash_value = id_parts[0]
            url = get_url_from_sqlite(hash_value)

            processed_results.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata'],
                'url': url,
                'text': match['metadata'].get('window', 'No text available')
            })

    return processed_results

# Main loop for multiple queries
while True:
    query_text = input("Enter your query text (or 'quit' to exit): ")
    if query_text.lower() == 'quit':
        break

    results = process_query(query_text)

    print(f"Query Text: {query_text}")
    print("---")

    for result in results:
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']}")
        print(f"Metadata: {result['metadata']}")
        print(f"URL: {result['url']}")
        print(f"Text: {result['text']}")
        print("---")



