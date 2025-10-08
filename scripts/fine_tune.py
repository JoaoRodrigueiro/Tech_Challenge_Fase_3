import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainingArguments, 
)


from peft import LoraConfig, get_peft_model, TaskType

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
TRAIN_PATH = DATA_DIR / "sft_train.jsonl"
VAL_PATH = DATA_DIR / "sft_val.jsonl"
OUT_DIR = BASE_DIR / "models" / "flan_t5_small_lora"

MODEL_NAME = "google/flan-t5-small"

MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 128
LR = 2e-4
BATCH_SIZE = 4          
GRAD_ACCUM = 4          
EPOCHS = 2              
WARMUP_RATIO = 0.03

LOG_STEPS = 50
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_LIMIT = 2

SEED = 42

def load_jsonl(path: Path, limit: int = None) -> List[Dict[str, str]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            ex = json.loads(line)

            if "prompt" in ex and "response" in ex:
                data.append(ex)
    if not data:
        raise RuntimeError(f"Nenhum dado lido de {path}")
    return data

def make_datasets():
    train_data = load_jsonl(TRAIN_PATH)
    val_data = load_jsonl(VAL_PATH)

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)
    return train_ds, val_ds

def main():
    torch.manual_seed(SEED)

    print("Carregando tokenizer e modelo base...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "k", "v", "o"],  
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds, val_ds = make_datasets()

    def preprocess(batch):
        inputs = tokenizer(
            batch["prompt"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            batch["response"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
    output_dir=str(OUT_DIR),
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_ratio=WARMUP_RATIO,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_strategy="steps",
    logging_steps=LOG_STEPS,

    save_total_limit=SAVE_LIMIT,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    predict_with_generate=True,   # agora OK, pois é Seq2SeqTrainingArguments
    remove_unused_columns=True,
    report_to=[],                 # em vez de "none"
    dataloader_num_workers=0,
    seed=SEED,
)


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Iniciando treino (LoRA em FLAN-T5-small)...")
    trainer.train()
    print("Treino finalizado.")

    print("Salvando adapter LoRA...")
    trainer.save_model()         # salva modelo+adapter
    tokenizer.save_pretrained(OUT_DIR)

    print(f"✅ Adapter salvo em: {OUT_DIR}")

if __name__ == "__main__":
    main()
