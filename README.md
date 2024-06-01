# generator-gpt2
# Simple NLP Model using Hugging Face

This repository contains a simple Natural Language Processing (NLP) model built using the Hugging Face Transformers library. The model is designed for text generation tasks and can be easily adapted for other NLP tasks such as text classification, named entity recognition, and more.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates how to create and use a simple NLP model for text generation using the Hugging Face Transformers library. Hugging Face provides state-of-the-art pre-trained models that can be fine-tuned for various NLP tasks.

## Installation

To use this project, you need to have Python installed. You can install the required packages using pip:

```bash
pip install transformers torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# Encode the input text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


# Encode the input text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
input_text = "The future of AI"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=3,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
