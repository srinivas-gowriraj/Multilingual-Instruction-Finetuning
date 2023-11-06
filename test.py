from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM
import argparse
from trl import SFTTrainer
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from evaluate import load
from vllm import LLM, SamplingParams
import pandas as pd



def load_tydiqa(language):
    # Load dataset
    dataset=load_dataset('tydiqa', 'secondary_task', split='validation') #tydiqa has no test set
    dataset = dataset.filter(lambda x: x["id"].startswith(language))
    #dataset = dataset.map(lambda x: {"prompt": f"Read the context and answer the question in one or a few words in English. ### Context: {x['context']} ### Question: {x['question']} ### Answer: "})
    dataset = dataset.map(lambda x:{'prompt': f"اقرأ السياق وأجب عن السؤال بكلمة واحدة أو بضع كلمات. ### السياق: {x['context']} ### السؤال: {x['question']} ### الإجابة: "})
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    



def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    set_seed(args)

    # Load dataset
    dataset=load_tydiqa(args.language)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, cache_dir="./hf_cache", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    #tokenize the dataset
    #dataset = dataset.map(lambda x: tokenizer(x["prompt"], truncation=True), batched=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir="./hf_cache", trust_remote_code=True)

    #model.to(device)
    # model.eval()
    generated_answers = []
    ground_truth_answers = []


    for i in tqdm(range(0,len(dataset))):
        prompt = dataset[i]["prompt"]
        # print(prompt)
        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, device=device)
        result = pipe(f"{prompt}")
        result = result[0]['generated_text'].replace(prompt, "").strip()
        result = result.split("###")[0].strip()
        generated_answers.append({"prediction_text": result, "id": dataset[i]["id"]})
        ground_truth_answers.append({"answers": dataset[i]["answers"], "id": dataset[i]["id"]})

        
    # print(generated_answers)
    # print(ground_truth_answers)

    squad_metric = load("squad")
    result = squad_metric.compute(predictions=generated_answers, references=ground_truth_answers)
    print(result)

    #save the generated answers and ground truth answers as csv
    
    df = pd.DataFrame()
    df["id"] = [x["id"] for x in generated_answers]
    df["prediction_text"] = [x["prediction_text"] for x in generated_answers]
    df["answers"] = [x["answers"] for x in ground_truth_answers]
    df.to_csv(f"results_multialpaca_{args.dataset_name}_{args.language}.csv", index=False)

        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", default="meta-llama/Llama-2-7b-hf", type=str, help="Base Model name")
    parser.add_argument("--model_name", default="llama-2-7B-multialpaca-hf/final_merged_checkpoint", type=str, help="Model name")
    parser.add_argument("--dataset_name", default="tydiqa", type=str, help="Dataset name")
    #parser.add_argument("--language", default="indonesian", type=str, help="Language")
    parser.add_argument("--language", default="arabic", type=str, help="Language")
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    args=parser.parse_args()
    main(args)