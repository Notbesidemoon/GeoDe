import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import argparse
import os
import ipdb
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

def load_model_and_tokenizer(model_name):
    """
    load model and tokenizer
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # set pad_token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def get_last_hidden_states(model, tokenizer, text, max_length=512):
    """
    get the hidden states of all layers of the last token of the text
    return a list, each element is the hidden state of the last token of each layer
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # get the hidden states of all layers of the last token of the text
    all_layers_last_token_hidden_states = []
    for layer_hidden_states in outputs.hidden_states:  # traverse all layers
        # layer_hidden_states: [batch_size, seq_len, hidden_size]
        last_token_hidden_state = layer_hidden_states[0, -1, :].cpu().numpy()  # get the last token
        all_layers_last_token_hidden_states.append(last_token_hidden_state)
  
    # ipdb.set_trace()
    
    return all_layers_last_token_hidden_states

def process_json_file(json_path, model_name, max_length=512):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"The data contains {len(data)} samples")
    model, tokenizer = load_model_and_tokenizer(model_name)
    data = data["samples"]
    # process each sample
    processed_data = []
    error_count = 0
    
    for i, sample in enumerate(tqdm(data, desc="Processing samples")):
        question = "Question: " + sample["question"] + " Answer: " + sample["low_t_ans"]
        new_sample = {"id": i, "question": question, "correct": sample["correct"], "question_hs": None}
        
        question_hidden_state = get_last_hidden_states(model, tokenizer, question, max_length)
        new_sample["question_hs"] = question_hidden_state
        processed_data.append(new_sample)

    output_pkl_path = json_path.replace(".json", "_qahs.pkl")
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    print(f"Saving to: {output_pkl_path}")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Successfully processed {len(processed_data)} samples")
    print(f"Error samples: {error_count}")
    return processed_data

def main():
    parser = argparse.ArgumentParser(description='Process JSON file and add LLM hidden states')
    parser.add_argument('--input_json', type=str, required=True, 
                       help='The input JSON file path')
    parser.add_argument('--model_name', type=str, 
                      required=True,
                       help='The model name to use')
    parser.add_argument('--max_length', type=int, default=512,
                       help='The maximum sequence length')
    parser.add_argument('--gpu', type=int, required=True,
                       help='The GPU number to use')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # process data
    processed_data = process_json_file(
        args.input_json, 
        args.model_name,
        args.max_length
    )
    
    # verify results
    print(f"\nProcessing completed!")
    print(f"Original data: {args.input_json}")
    print(f"Output file: {args.input_json.replace('.json', '_hs.pkl')}")
    print(f"Processed samples: {len(processed_data)}")

if __name__ == "__main__":
    main()