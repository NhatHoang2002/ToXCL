import csv
import os
import sys
import time
from argparse import ArgumentParser

import openai
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eval_metrics import (compute_classification_scores,
                          compute_generation_scores)
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

openai.api_key = "<YOUR_OPENAI_API_KEY}"
device = "cuda"
model, tokenizer = None, None

PROMPT_TEMPLATE = """
The input is a tweet that might contain toxic speech. You are required to detect whether it is hateful or not, and if it is hateful, please give a brief explanation of why it is considered hateful.
The Output can be either of the form "hate <SEP> <GENERATED_EXPLANATION>" or "normal <SEP> none".

Input: {}
Output:
"""

def get_mistral_answer(prompt):
    messages = [{"role": "user", "content": prompt}]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=32, do_sample=True)
    assistant_message = tokenizer.batch_decode(generated_ids)[0]
    return assistant_message.split("[/INST]")[1].strip().split("</s>")[0].strip()

def get_chatgpt_answer(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages
            )
        except Exception as msg:
            print(msg)
            print('sleeping because of exception ...')
            time.sleep(30)
    response = response.choices[0].message["content"]
    return response

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('model_name', choices=['chatgpt', 'mistral'])
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--output_dir', default='../saved/llm')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset_name = args.test_data.split('/')[-1].split('_')[0]
    output_path = os.path.join(args.output_dir, f"{args.model_name}_{dataset_name}_result.csv")

    if args.model_name == 'chatgpt':
        get_output_fn = get_chatgpt_answer
    elif args.model_name == 'mistral':
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        get_output_fn = get_mistral_answer

    test_data = []
    with open(args.test_data, 'r', newline='') as file:
        csvreader = csv.reader(file)
        _ = next(csvreader)
        for row in csvreader:
            test_data.append({
                'text': row[0].strip(),
                'label': row[2].strip(),
                'explanation': row[3].strip(),
            })

    test_output = []
    for idx, row in tqdm(enumerate(test_data), total=len(test_data)):
        tqdm.write(f">>> {row['text']}")
        prompt = PROMPT_TEMPLATE.format(row['text'])
        output = get_output_fn(prompt)

        test_output.append({
            'text': row['text'],
            'output': output,
            'label': row['label'],
            'gold_explanation': row['explanation']
        })
        pd.DataFrame(test_output).to_csv(output_path, index=False)

    df = pd.read_csv(output_path)
    cls_ground_truth = df.label.tolist()
    generation_ground_truth = df.gold_explanation.tolist()

    output = df.output.tolist()
    cls_generated, generation_generated = [], []
    for x in output:
        if "<SEP>" in x:
            cls_generated.append(x.split("<SEP>")[0].strip())
            generation_generated.append(x.split("<SEP>")[1].strip())
        else:
            cls_generated.append("hate")
            generation_generated.append(x)

    print("Classification (ACC, F1):", compute_classification_scores(cls_ground_truth, cls_generated))
    print("Generation (BLEU-4, ROUGE-L, METEOR, BERTScore):", compute_generation_scores(generation_ground_truth, generation_generated))
