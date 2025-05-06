from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
from dataset.preprocessor import prepare_dataset
from utils import save_lora_adapter
import torch
import os
import argparse
import logging

#other file imports

#LOAD IN 8 BIT?

def train(model_name, dataset_path, output_dir, config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        r = config.get("r", 8),
        lora_alpha = config.get("lora_alpha", 16),
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout = config.get("lora_dropout", 0.05),
        bias = config.get("bias", "none"),
        task_type = TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    dataset = prepare_dataset(dataset_path, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        num_train_epochs=config.get("num_train_epochs", 3),
        logging_dir=os.path.join(output_dir, "logs"),
        fp16=True,
        logging_steps=config.get("logging_steps", 10),
        save_strategy=config.get("save_strategy", "epoch"),
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    save_lora_adapter(model, output_dir)

