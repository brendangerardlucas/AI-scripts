#Use or modify this flask app to run LLMs in your website
#change model_name to access different models from huggingface
#!/usr/bin/env python3
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})  # Add this line to enable CORS for all routes and origins

def initialize_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Add this line to set padding_side to 'left'
    return model, tokenizer

def generate_response(model, tokenizer, chat_history_text):
    encoded_input = tokenizer.encode(chat_history_text, return_tensors="pt", truncation=True, max_length=175)
    attention_mask = (encoded_input != tokenizer.pad_token_id).float()
    response = model.generate(encoded_input, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask)
    decoded_response = tokenizer.decode(response[:, encoded_input.shape[-1]:][0], skip_special_tokens=True)
    return decoded_response

app = Flask(__name__)

model_name = "cerebras/Cerebras-GPT-111M"
model, tokenizer = initialize_model(model_name)

@app.route('/generate', methods=['POST'])
@cross_origin()
def generate():
    data = request.get_json()
    
    if not data or 'chat_history_text' not in data or not data['chat_history_text']:
        return jsonify({'response': 'Invalid input'})

    chat_history_text = data['chat_history_text']
    print(f"Received input: {chat_history_text}")  # Print input text
    response = generate_response(model, tokenizer, chat_history_text)
    print(f"Generated response: {response}")  # Print generated response
    return jsonify({'id': 'dummy-id', 'message': response})  # Changed this line to include 'id' and 'message' fields

@app.route('/ping', methods=['GET'])
@cross_origin()
def ping():
    print('Connected')
    return jsonify({'response': 'Connected'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
