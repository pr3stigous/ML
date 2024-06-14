#code below was used in google colab to fine-tune a small FLAN-T5

%pip install --upgrade pip
%pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1 --quiet

%pip install \
    transformers==4.27.2 \
    datasets==2.11.0

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration
from google.colab import drive
import torch
import time
import pandas as pd
import numpy as np

model_name='google/flan-t5-small'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# it is important for your dataset to have both train and validation data
# if you don't have validation data, just split up your data into an 85/15 split

huggingface_dataset_name = "" # your dataset from huggingface

dataset = load_dataset(huggingface_dataset_name)

dataset

# i had used code below to tune a model to be an AI fitness coach - hence the prompt

def tokenize_function(example):
    start_prompt = 'You are an empathetic physical fitness AI assistant. You want to educate others about the '
    end_prompt = ' exercise'
    prompt = [start_prompt + exercise_name + end_prompt for exercise_name in example["name"]]
    example['input_ids'] = tokenizer(prompt, max_length=256, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["output"], max_length=256, padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example

# Specify the desired batch size
desired_batch_size = 32  # Change this to your preferred batch size

# The dataset actually contains 2 diff splits: train and validation.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=desired_batch_size)
tokenized_datasets = tokenized_datasets.remove_columns(['name', 'force', 'mechanic', 'equipment', 'primaryMuscles', 'secondaryMuscles', 'instructions', 'output'])

class SaveModelCallback:
    def __init__(self):
        pass

    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_hyper_param_search:
            return
        # Explicitly save the model after training
        torch.save(trainer.model.state_dict(), 'yourmodelfilename.pth') 

    # Placeholder method to satisfy Trainer's requirements
    def on_prediction_step(self, args, state, control, **kwargs):
        pass

    # Placeholder method to satisfy Trainer's requirements
    def on_evaluate(self, args, state, control, **kwargs):
        pass

    # Placeholder method to satisfy Trainer's requirements
    def on_save(self, args, state, control, **kwargs):
        pass


# Create an instance of the SaveModelCallback
# this allows you to save a model to your environment
save_model_callback = SaveModelCallback()


# below is another hyperperameter that you can use for tuning
# the second line below needs to be added to training_args for it to function
"""
gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches before updating parameters
    gradient_accumulation_steps=gradient_accumulation_steps  # Set gradient accumulation steps
"""

# Set up training arguments
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=200,
    weight_decay=0.01,
    evaluation_strategy="epoch"  # Evaluation after each epoch
)

# Set up Trainer
trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[save_model_callback]
)

trainer.train()

# the below is to save your model to google drive
# this is much faster than just downloading the model that is saved to your environment

drive.mount('/content/drive')

# the actual saving process
!cp /content/yourmodelfilename.pth /content/drive/My\ Drive/

# askng the model questions
# i had initially loaded the model i tuned into the same environment - hence the addition of '2' to tokenizer and model

# Check if GPU is available and set appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model state dictionary from Google Drive, specifying the device

model_state_dict = torch.load('/content/yourmodelfilename.pth', map_location=device)

# Instantiate the tokenizer
tokenizer2 = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Instantiate the T5 model architecture
model2 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


# Load the saved model weights
model2.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model2.eval()

# printing out parameter information

# Count trainable model parameters
trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)

# Count all model parameters
total_params = sum(p.numel() for p in model2.parameters())

# Calculate percentage of trainable model parameters
percentage_trainable_params = (trainable_params / total_params) * 100

# Print out the information
print("Trainable model parameters:", trainable_params)
print("All model parameters:", total_params)
print("Percentage of trainable model parameters:", percentage_trainable_params, "%")

# Function to generate text from the model
def generate_text(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model2.generate(input_ids, max_length=100, num_beams=10, do_sample=True, temperature=1.0)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "what is a physical exercise?"
generated_response = generate_text(prompt)
print(generated_response)

#Here's a version that takes a list of prompts and returns a list of generated
#responses:

# Function to generate text from the model
def generate_text(prompts, num_outputs=1):
    # Tokenize all prompts together
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids
    # Generate outputs
    outputs = model2.generate(input_ids, max_length=100, num_beams=10, num_return_sequences=num_outputs, temperature=1.0, do_sample=True)
    # Decode generated sequences
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

# Example usage
prompts = ["what is a physical exercise?", "how to stay fit at home?", "importance of stretching before exercise"]
generated_responses = generate_text(prompts, num_outputs=3)
for prompt, response_set in zip(prompts, generated_responses):
    print("Prompt:", prompt)
    for i, response in enumerate(response_set):
        print(f"Generated response {i+1}:", response)
    print()

#this version shows you which output ties to which prompt

# Function to generate text from the model
def generate_text(prompts, num_outputs=1):
    # Tokenize all prompts together
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids
    # Generate outputs
    outputs = model2.generate(input_ids, max_length=100, num_beams=10, num_return_sequences=num_outputs, temperature=1.0, do_sample=True)
    # Decode generated sequences
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

# Example usage
prompts = ["what is a physical exercise?", "how to stay fit at home?", "importance of stretching before exercise"]
generated_responses = generate_text(prompts, num_outputs=3)

# Print prompts and their corresponding generated responses
for i, prompt in enumerate(prompts):
    print("Prompt:", prompt)
    for j, response in enumerate(generated_responses[i]):
        print(f"Generated response {j+1}:", response)
    print()

#for testing multiple models at once

# Function to generate text from the models
def generate_text(prompts, models, num_outputs=1):
    generated_responses = []
    for model in models:
        # Tokenize all prompts together for the current model
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids
        # Generate outputs for the current model
        outputs = model.generate(input_ids, max_length=100, num_beams=10, num_return_sequences=num_outputs, temperature=1.0, do_sample=True)
        # Decode generated sequences
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_responses.append(generated_texts)
    return generated_responses

# Example usage with three different FLAN-T5 models
models = [model1, model2, model3]  # Assuming you have three different FLAN-T5 models
prompts = ["what is a physical exercise?", "how to stay fit at home?", "importance of stretching before exercise"]
generated_responses = generate_text(prompts, models, num_outputs=3)

# Print prompts and their corresponding generated responses for each model
for i, prompt in enumerate(prompts):
    print("Prompt:", prompt)
    for j, model_responses in enumerate(generated_responses):
        print(f"Model {j+1} generated responses:")
        for k, response in enumerate(model_responses[i]):
            print(f"  Generated response {k+1}:", response)
    print()

#if you want the above to be outputted as a dataframe instead

import pandas as pd

# Function to generate text from the models
def generate_text(prompts, models, num_outputs=1):
    generated_responses = []
    for model in models:
        # Tokenize all prompts together for the current model
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids
        # Generate outputs for the current model
        outputs = model.generate(input_ids, max_length=100, num_beams=10, num_return_sequences=num_outputs, temperature=1.0, do_sample=True)
        # Decode generated sequences
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_responses.append(generated_texts)
    return generated_responses

# Example usage with three different FLAN-T5 models
models = [model1, model2, model3]  # Assuming you have three different FLAN-T5 models
prompts = ["what is a physical exercise?", "how to stay fit at home?", "importance of stretching before exercise"]
generated_responses = generate_text(prompts, models, num_outputs=3)

# Create a DataFrame to store the results
data = []
for i, prompt in enumerate(prompts):
    for j, model_responses in enumerate(generated_responses):
        for k, response in enumerate(model_responses[i]):
            data.append({"Prompt": prompt, "Model": f"Model {j+1}", "Generated Response": response})

df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# and if you want the prompts from above to be provided via a txt file

import pandas as pd

# Function to read prompts from a text file
def read_prompts_from_file(file_path):
    with open(file_path, "r") as file:
        prompts = [line.strip() for line in file]
    return prompts

# Function to generate text from the models
def generate_text(prompts, models, num_outputs=1):
    generated_responses = []
    for model in models:
        # Tokenize all prompts together for the current model
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids
        # Generate outputs for the current model
        outputs = model.generate(input_ids, max_length=100, num_beams=10, num_return_sequences=num_outputs, temperature=1.0, do_sample=True)
        # Decode generated sequences
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_responses.append(generated_texts)
    return generated_responses

# Example usage with three different FLAN-T5 models
models = [model1, model2, model3]  # Assuming you have three different FLAN-T5 models
prompts_file = "prompts.txt"  # Path to the text file containing prompts
prompts = read_prompts_from_file(prompts_file)
generated_responses = generate_text(prompts, models, num_outputs=3)

# Create a DataFrame to store the results
data = []
for i, prompt in enumerate(prompts):
    for j, model_responses in enumerate(generated_responses):
        for k, response in enumerate(model_responses[i]):
            data.append({"Prompt": prompt, "Model": f"Model {j+1}", "Generated Response": response})

df = pd.DataFrame(data)

# Display the DataFrame
print(df)
