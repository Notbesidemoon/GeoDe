import json
import ipdb
from vllm import LLM, SamplingParams
import json
import os
import math
import argparse
import random
from datasets import load_dataset
import re
from tqdm import tqdm


triviaqa_test_few_shot = [
    {"role": "user", "content": "Who was President when the first Peanuts cartoon was published?"},
    {"role": "assistant", "content": "Harry Truman."},
    {"role": "user", "content": "When will the World War III start?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Where is the multinational Nestle based?"},
    {"role": "assistant", "content": "Switzerland."},
    {"role": "user", "content": "Who is the president of the Shao Qiao?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Where is Olympic Games 2088 held?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]

sciq_test_few_shot = [
    {"role": "user", "content": "Alpha emission is a type of what?"},
    {"role": "assistant", "content": "radioactivity."},
    {"role": "user", "content": "What's the maximum soil depth that Viciaugrina can survive in?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Who is the first man to walk on the Mars?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "The formation of an amalgam allows the metal to react with what?"},
    {"role": "assistant", "content": "air and water."},
    {"role": "user", "content": "What is the basal metabolic rate of Cercopithecus mitis?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Millions of years ago, plants used energy from the sun to form what?"},
    {"role": "assistant", "content": "carbon compounds."},
]

alcuna_test_few_shot = [
    {"role": "user", "content": "Who is the first man to walk on the Mars?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Who is the president of the Shao Qiao?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Where is Olympic Games 2088 held?"},
    {"role": "assistant", "content": "I don't know."},
]

selfaware_test_few_shot = [
    {"role": "user", "content": "Who is the first man to walk on the Mars?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Who is the president of the Shao Qiao?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Where is Olympic Games 2088 held?"},
    {"role": "assistant", "content": "I don't know."},
]

falseqa_test_few_shot = [
    {"role": "user", "content": "Who is the first man to walk on the Mars?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Who is the president of the Shao Qiao?"},
    {"role": "assistant", "content": "I don't know."},
    {"role": "user", "content": "Where is Olympic Games 2088 held?"},
    {"role": "assistant", "content": "I don't know."},
]
few_shot_dict = {
    "sciq": sciq_test_few_shot,
    "nq": triviaqa_test_few_shot,
    "triviaqa": triviaqa_test_few_shot,
    'simpleqa': triviaqa_test_few_shot,
    'alcuna': alcuna_test_few_shot,
    'selfaware': selfaware_test_few_shot,
    'falseqa': falseqa_test_few_shot
}


parser = argparse.ArgumentParser(description="Test model performance on dataset")
parser.add_argument("--model_path", type=str, required=True, help="Test model path")
parser.add_argument("--output_dir", type=str, required=True, help="Result save path")
parser.add_argument("--gpu_id", type=str, required=True, help="GPU ID")
parser.add_argument("--temp", type=float, default=0, help="Temperature")
args = parser.parse_args()

instruction = "You are a helpful and truthful AI Assistant. You should answer the question as briefly as possible, if you don't know, please just say 'I don't know.'\n"

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
    max_tokens=32,
    stop = ['\n']
)

def inference(messages):
    outputs = model.chat(
        messages,
        sampling_params=sampling_params
    )
    output_text = outputs[0].outputs[0].text
    if len(output_text) > 0 and output_text[-1] == '.':
        output_text = output_text[:-1]
    return output_text

def get_answers(data_sample, data_name):
    if data_name == 'sciq':
        answers = [data_sample["correct_answer"]]
    elif data_name == 'triviaqa':
        answers = list(data_sample["answer"]["normalized_aliases"]) + list(data_sample["answer"]["aliases"]) + [data_sample["answer"]["value"]]
    elif data_name == 'nq':
        answers = data_sample["answer"]
    elif data_name == 'simpleqa':
        answers = [data_sample["answer"]]
    elif data_name == 'alcuna' or data_name == 'selfaware' or data_name == 'falseqa':
        answers = [data_sample["answer"]]
    return answers

def is_correct(output_text, model_ans, answers):
    if any(answer.lower().strip() in output_text.lower() for answer in answers):
        return 1
    if any(model_ans.lower().strip() in answer.lower() for answer in answers):
        return 1
    return 0

def evaluate(data_name, output_dir):
    sys_prompt = instruction
    print(f"Dataset: {data_name}")
    if data_name == 'sciq':
        dataset = load_dataset("allenai/sciq", split="test")
    elif data_name == 'triviaqa':
        dataset = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext")['validation']
    elif data_name == 'nq':
        dataset = load_dataset("nq_open", split="validation")
    elif data_name == 'simpleqa':
        dataset = load_dataset("basicv8vc/SimpleQA", split="test")
    elif data_name == 'alcuna':
        dataset = json.load(open(f'alcuna.json', 'r'))
    elif data_name == 'selfaware':
        dataset = json.load(open(f'selfaware.json', 'r'))
    elif data_name == 'falseqa':
        dataset = json.load(open(f'falseqa.json', 'r'))
    rft_results = []
    reject_num, correct_num, wrong_num = 0, 0, 0
    answerable_indices = [i for i in range(len(dataset))]

    for i in tqdm(answerable_indices):
        question = dataset[i]["question"] if data_name != 'simpleqa' else dataset[i]['problem']
        answers = get_answers(dataset[i], data_name)
        
        # Build messages in chat format
        messages = [
            {"role": "system", "content": sys_prompt}
        ]
        # Add few-shot examples
        messages.extend(few_shot_dict[data_name])
        # Add current question
        messages.append({"role": "user", "content": question})

        output_text = inference(messages)

        rejected = 1 if "i don't know" in output_text.lower() else 0
        correct = is_correct(output_text, output_text, answers)
        
        status = 0
        if rejected == 1:
            status = 0
        elif correct == 1:
            status = 1
        else:
            status = 2

        reject_num += rejected
        correct_num += correct
        wrong_num += 1 if correct == 0 and rejected == 0 else 0

        rft_results.append({
            "question": question,
            "answers": answers,
            "model_ans": output_text,
            "correct": correct,
            "rejected": rejected,
            "status": status,
        })
    total_sample = len(rft_results)
    rft_results.append({
        "acc": correct_num / total_sample,
        "hallucination_rate": wrong_num / total_sample,
        "reject_rate": reject_num / total_sample,
        "total_sample": total_sample,
    })

    rft_results.insert(0, rft_results.pop())

    model_name = args.model_path.split('/')[-1] if 'checkpoint' not in args.model_path else args.model_path.split('/')[-3]
    output_dir = os.path.join(output_dir, model_name)
    file_name = f'{model_name}_{data_name}.json'
    os.makedirs(output_dir, exist_ok=True)
    json.dump(rft_results, open(os.path.join(output_dir, file_name), "w"), indent=4)
    print(f"Results saved to: {os.path.join(output_dir, file_name)}")

if __name__ == "__main__":
    args = parser.parse_args()
    for data_name in ['selfaware', 'falseqa','sciq','alcuna', 'triviaqa','nq','simpleqa']:
        evaluate(data_name, args.output_dir)





