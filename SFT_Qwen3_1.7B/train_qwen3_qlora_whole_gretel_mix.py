import os
import sys
import torch
from datasets import load_dataset
from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

def main():
    from huggingface_hub import login
    login("XXX")  # API key redacted

    if not torch.cuda.is_available():
        print("no cuda")
        sys.exit(1)

    BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
    OUTPUT_DIR = "Qwen3-1.7B_safety_mix_qlora"

    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj"]

    MAX_PROMPT_LENGTH = 500
    MAX_TARGET_LENGTH = 500
    MAX_COMBINED = MAX_PROMPT_LENGTH + MAX_TARGET_LENGTH

    TRAIN_EPOCHS = 3
    PER_DEVICE_BATCH_SIZE = 1    
    GRADIENT_ACCUMULATION_STEPS = 8 
    LEARNING_RATE = 2e-4
    FP16 = True
    LOGGING_STEPS = 50
    SAVE_STEPS = 100
    SAVE_TOTAL_LIMIT = 5

    df_local = pd.read_excel("dataset/gretelai_prompt_response_mix.xlsx", usecols=["prompt", "safe_response"])
    df_local["row_idx"] = df_local.index

    ds_full = Dataset.from_pandas(df_local)
    print("loaded Gretel dataset")

    def make_example(example):
        instr = example["prompt"].strip()
        safe  = example["safe_response"].strip()
        idx   = example["row_idx"]
        if idx % 2 == 0:
            input_text = f"Instruction: {instr}\nAnswer: "
        else:
            input_text = f"Istruzione: {instr}\nRisposta: "
        return {
            "input_text":  input_text,
            "target_text": safe
        }

    ds_tuning = ds_full.map(
        make_example,
        remove_columns=["prompt", "safe_response", "row_idx"]
    )

    print(ds_tuning[0])
    print(ds_tuning[1])
    print(ds_tuning[2])
    print(ds_tuning[3])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id 

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=False, 
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto", 
        token=True,
    )

    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)

    def count_params(m):
        trainable, total = 0, 0
        for p in m.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        return trainable, total

    t_params, total_params = count_params(model)
    print(f"{t_params} / {total_params} trainable parameters")

    def tokenize_fn(examples):
        enc = tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=MAX_PROMPT_LENGTH,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            tgt = tokenizer(
                examples["target_text"],
                truncation=True,
                max_length=MAX_TARGET_LENGTH,
                padding="max_length",
            )

        input_ids_batch = []
        attention_masks_batch = []
        labels_batch = []

        for in_ids, tgt_ids in zip(enc["input_ids"], tgt["input_ids"]):
            combined = in_ids + tgt_ids
            labels = [-100] * len(in_ids) + tgt_ids

            combined = combined[:MAX_COMBINED]
            labels = labels[:MAX_COMBINED]

            mask = [1] * len(combined)
            pad_len = MAX_COMBINED - len(combined)
            if pad_len > 0:
                combined += [tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
                mask += [0] * pad_len

            input_ids_batch.append(combined)
            labels_batch.append(labels)
            attention_masks_batch.append(mask)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks_batch,
            "labels": labels_batch,
        }

    ds_train = ds_tuning.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds_tuning.column_names,
    )
    ds_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    print(f"adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
