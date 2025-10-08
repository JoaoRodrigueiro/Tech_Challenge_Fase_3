import gzip
import json
import os
import itertools
from pathlib import Path
from typing import Iterator, Dict
import html

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "trn.json.gz"  
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = PROC_DIR / "sft_train.jsonl"
VAL_OUT = PROC_DIR / "sft_val.jsonl"


SAMPLE_SIZE = 2000  
VAL_RATIO = 0.05     

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)   # <— decodifica entidades HTML
    s = s.strip()
    s = " ".join(s.split())
    return s

def make_prompt(title: str) -> str:
    return (
        "Dado o TÍTULO de um produto da Amazon, responda apenas com a sua DESCRIÇÃO oficial.\n"
        f"Título: \"{title}\"\n"
        "Descrição:"
    )

def stream_jsonl_gz(path: Path) -> Iterator[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {RAW_PATH}\n"
            f"Coloque o trn.json.gz em: {RAW_PATH}"
        )
    
    records = []
    for ex in itertools.islice(stream_jsonl_gz(RAW_PATH), SAMPLE_SIZE * 2):
        title = clean_text(ex.get("title", ""))
        content = clean_text(ex.get("content", ""))
        uid = str(ex.get("uid", "")).strip()   # <— novo
        if len(title) < 3 or len(content) < 5:
            continue
        records.append((uid, title, content))  # <— guarda uid junto
        if len(records) >= SAMPLE_SIZE:
            break

    if not records:
        raise RuntimeError("Nenhum registro válido encontrado (title/content vazios).")

    n_total = len(records)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val

    train = records[:n_train]
    val = records[n_train:]

    with open(TRAIN_OUT, "w", encoding="utf-8") as ft:
        for uid, title, content in train:
            obj = {
                "prompt": make_prompt(title),
                "response": content,
                "title": title,     
                "uid": uid          
            }
            ft.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(VAL_OUT, "w", encoding="utf-8") as fv:
        for uid, title, content in val:
            obj = {
                "prompt": make_prompt(title),
                "response": content,
                "title": title,      
                "uid": uid           
            }
            fv.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("✅ Sample processado com sucesso!")
    print(f"Total: {n_total} | train: {n_train} | val: {n_val}")
    print(f"Arquivos:")
    print(f" - {TRAIN_OUT}")
    print(f" - {VAL_OUT}")

if __name__ == "__main__":
    main()
