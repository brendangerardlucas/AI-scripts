#use this to chat with LLMs in your terminal
#change model_name to use different models from the huggingface transformers library

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)

def chat_with_model(model, tokenizer):
    print("You can start chatting now (type 'quit' to exit):")
    tokenizer.pad_token = tokenizer.eos_token
    chat_history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break

        chat_history.append(f"User: {user_input}")
        chat_history_text = ' '.join(chat_history)

        encoded_input = tokenizer.encode(chat_history_text, return_tensors="pt", padding='max_length', max_length=32)
        attention_mask = (encoded_input != tokenizer.pad_token_id).float()
        response = model.generate(encoded_input, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask)
        decoded_response = tokenizer.decode(response[:, encoded_input.shape[-1]:][0], skip_special_tokens=True)

        print(f"Model: {decoded_response}")
        chat_history.append(f"Model: {decoded_response}")

model_name = "cerebras/Cerebras-GPT-111M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

chat_with_model(model, tokenizer)
