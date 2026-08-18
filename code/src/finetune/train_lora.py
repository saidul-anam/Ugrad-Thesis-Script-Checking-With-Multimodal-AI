"""
Kaggle-compatible LoRA / QLoRA fine-tuning script using PEFT and TRL's SFTTrainer.
Supports checkpointing to /kaggle/working, resume capability, and adapter saving.
"""

import os
import argparse
from typing import Optional


def run_lora_finetune(
    train_jsonl: str,
    base_model_name: str = "google/gemma-3-27b-it",
    output_dir: str = "/kaggle/working/lora_checkpoints",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    resume_from_checkpoint: Optional[str] = None
) -> None:
    """
    Launch QLoRA fine-tuning job with PEFT and SFTTrainer.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    print(f"Loading base model: {base_model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    dataset = load_dataset("json", data_files=train_jsonl, split="train")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="messages",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args
    )

    print("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final LoRA adapter weights
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"LoRA Adapter saved to {final_adapter_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning for Grading Pipeline")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to training jsonl")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-27b-it", help="Base model identifier")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/lora_checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    run_lora_finetune(
        train_jsonl=args.train_jsonl,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs
    )
