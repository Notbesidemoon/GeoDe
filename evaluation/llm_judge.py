import json
import ipdb
import torch


from vllm import LLM, SamplingParams
import json
import os
import math
import argparse
import random
from datasets import load_dataset
import re
from tqdm import tqdm
import torch

# LLM Judge few-shot examples
JUDGE_FEW_SHOT_EXAMPLES = [
    {
        "question": "What three parts is the earth divided into?", 
        "ground_truth": "crust, mantle, core",
        "proposed_answer": "1. Crust, 2. Mantle, 3. Core",
        "judge_response": "yes"
    },
    {
        "question": "What is 2+2?",
        "ground_truth": "4",
        "proposed_answer": "The answer is 5.",
        "judge_response": "no"
    },
    {
        "question": "What is the most abundant metal of the earth's crust?",
        "ground_truth": "aluminum",
        "proposed_answer": "Aluminium",
        "judge_response": "yes"
    },
    {
        "question": "What do metals start out as?",
        "ground_truth": "ore",
        "proposed_answer": "ionic compounds",
        "judge_response": "no"
    },
    {
        "question": "What color is the sky?",
        "ground_truth": "blue",
        "proposed_answer": "The sky is green.",
        "judge_response": "no"
    },
    {
        "question": "Trees and shrubs are example of what type of plant?",
        "ground_truth": "perennials",
        "proposed_answer": "Woody plants",
        "judge_response": "yes"
    },
]


def create_judge_prompt(question, ground_truth, proposed_answer):
    """Create LLM judge prompt"""
    prompt = "We are assessing the quality of answers to the following question: "
    prompt += f"{question}\n\n"
    prompt += f"The following are expected answers to this question: {ground_truth}\n\n"
    prompt += f"The proposed answer is: {proposed_answer}\n\n"
    prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?\n"
    prompt += "Respond only with yes or no.\n\n"
    
    # Shuffle judge few-shot examples each time
    judge_examples = JUDGE_FEW_SHOT_EXAMPLES
    # random.shuffle(judge_examples)
    
    # Add few-shot examples
    prompt += "Here are some examples:\n\n"
    for example in judge_examples:
        prompt += f"Question: {example['question']}\n"
        prompt += f"Expected answer: {example['ground_truth']}\n"
        prompt += f"Proposed answer: {example['proposed_answer']}\n"
        prompt += f"Response: {example['judge_response']}.\n\n"
    
    prompt += "Now evaluate the following:\n"
    prompt += f"Question: {question}\n"
    prompt += f"Expected answer: {ground_truth}\n"
    prompt += f"Proposed answer: {proposed_answer}\n"
    prompt += "Response:"
    
    return prompt

# Parse command line arguments
parser = argparse.ArgumentParser(description="LLM rejudge answer")
parser.add_argument("--model_path", type=str, default="llama3_1-8b-instruct", help="Model path")
parser.add_argument("--data_path", type=str, required=True, help="Data path")
parser.add_argument("--gpu_id", type=str, required=True, help="GPU ID")
parser.add_argument("--temp", type=float, default=0, help="Temperature")
args = parser.parse_args()


# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print(f"Using GPU: {args.gpu_id}")
print(f"Testing model: {args.model_path}")

# Load model
model = LLM(
    model=args.model_path,
    gpu_memory_utilization=0.9,
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=args.temp,
    max_tokens=16,
    stop = ['\n','.','?','!']
)

def inference(input_text):
    outputs = model.generate(
        input_text,
        sampling_params=sampling_params
    )
    output_text = outputs[0].outputs[0].text
    if len(output_text) > 0 and output_text[-1] == '.':
        output_text = output_text[:-1]
    return output_text


def llm_re_evaluate(data_path):
    data = json.load(open(data_path, 'r'))
    llmjudge_correct_num, llmjudge_wrong_num = 0, 0
    for i, sample in tqdm(enumerate(data[1:])):
       
        question = sample["question"]
        ground_truth = sample["answers"][-1] if "answers" in sample else sample["answer"]
        proposed_answer = sample["model_ans"]
        llmjudge_correct = sample["correct"]
        
        judge_prompt = create_judge_prompt(question, ground_truth, proposed_answer)
        judge_response = inference(judge_prompt)
        data[i+1]["llmjudge_response"] = judge_response.strip().lower()

        if "yes" in judge_response.strip().lower():
            llmjudge_correct = 1
        else:
            llmjudge_correct = 0

        data[i+1]["llmjudge_correct"] = llmjudge_correct
        if llmjudge_correct == 1:
            llmjudge_correct_num += 1
        else:
            llmjudge_wrong_num += 1
        llmjudge_status = 1 if llmjudge_correct == 1 else 2
        if sample["rejected"] == 1:
            llmjudge_status = 0
        data[i+1]["llmjudge_status"] = llmjudge_status

    total_sample = len(data[1:])
    reject_num = sum([sample["rejected"] for sample in data[1:]])
    data[0].update({
        "llmjudge_acc": llmjudge_correct_num / total_sample,
        "llmjudge_hallucination_rate": (total_sample - reject_num - llmjudge_correct_num) / total_sample,
    })
    
    json.dump(data, open(data_path, 'w'), indent=4, ensure_ascii=False)
    print(f"Results saved to: {data_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    llm_re_evaluate(args.data_path)






