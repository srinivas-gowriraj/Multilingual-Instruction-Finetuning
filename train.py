from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM
import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import time
import torch
import os
    
def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 2:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Input:
            {input_text}
            
            ### Response:
            {response}
            '''
        else:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {instruction}
            
            ### Response:
            {response}
            '''
        if len(text.split())<150:
            output_text.append(text)

    return output_text


def format_arabic_prompts_func(examples):
    if examples['input'] == None:
        return {"text": f"يوجد أدناه تعليمات تصف المهمة. اكتب الرد الذي يكمل الطلب بشكل مناسب. ### التعليمات: {examples['instruction']} ### الاستجابة: {examples['output']}"}
    else:
        return {"text": f"يوجد أدناه تعليمات تصف مهمة، مقترنة بإدخال يوفر سياقًا إضافيًا. اكتب الرد الذي يكمل الطلب بشكل مناسب. ### التعليمات: {examples['instruction']} ### الإدخال: {examples['input']} ### الاستجابة: {examples['output']}"}


def main(args):
    if args.dataset_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    elif args.dataset_name == "multialpaca":
        dataset = load_dataset('json',data_files="multialpaca/ar.jsonl", split="train")
        dataset = dataset.map(format_arabic_prompts_func)
    
    
    if args.model_name == "llama-2-7B":
        model_name = "meta-llama/Llama-2-7b-hf"

    
    
    #define the lora config
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05

    t0=time.time()

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        cache_dir="./hf_cache",
        device_map={"":0}
    )
   
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./hf_cache", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    print(f"Loading model and tokenizer took {time.time() - t0} seconds")


    #define the training arguments

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    #define the trainer arguments
    num_train_epochs = 3
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 128
    optim = "adamw_torch_fused"
    lr_scheduler_type = "cosine"
    #max_steps = (50000*3)//1024
    max_steps = -1
    warmup_ratio = 0.03
    save_steps = 0
    learning_rate = 3e-4
    weight_decay = 0.001
    logging_steps = 25
    fp16 = False
    bf16 = False
    max_grad_norm = 0.3
    group_by_length = True

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Maximum sequence length to use
    max_seq_length = 512

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # response_template_with_context = "### Response:"  # We added context here: "\n". This is enough for this tokenizer
    # response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

    # collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        #formatting_func=formatting_prompts_func,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    trainer.train()
    trainer.model.save_pretrained('llama-2-7B-multialpaca-hf')

    del model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained("llama-2-7B-multialpaca-hf", device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join("llama-2-7B-multialpaca-hf", "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)






    

    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="llama-2-7B", type=str, help="Model name")
    parser.add_argument("--dataset_name", default="multialpaca", type=str, help="Dataset name")
    parser.add_argument("--output_dir", default="./output", type=str, help="Output directory")
    args = parser.parse_args()
    main(args)