import argparse
import os

import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        if example['type'][i] == "mcqa":
            text = f"### 질문: {example['question'][i]}\n### 선택지:\n{example['options'][i]}\n### 정답: {example['reasoning_process'][i]}### 정답: {example['answer'][i]}"
        else:
            text = f"### 질문: {example['question'][i]}\n### 정답: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts


def main(args):
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_accumulation_steps=1,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        report_to="wandb",
        deepspeed=args.deepspeed_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = args.max_length

    train_dataset = load_dataset('json', data_files=args.train_data_path, split="train")
    eval_dataset = load_dataset('json', data_files=args.eval_data_path, split="train")

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config._attn_implementation = 'sdpa'

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 config=config,
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r,
                             lora_alpha=args.lora_alpha)
    peft_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    for name, param in model.named_parameters():
        param.data = param.data.contiguous()

    response_template = "### 정답: "
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(model=model,
                         args=training_args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         data_collator=collator,
                         max_seq_length=args.max_length,
                         formatting_func=formatting_prompts_func)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--eval_data_path", type=str)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--deepspeed_config", type=str, default="config/ds_zero3.json")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args, _ = parser.parse_known_args()
    main(args)
