import json
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
from peft import PeftModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"
TRAIN_PATH = PROC_DIR / "sft_train.jsonl"
VAL_PATH = PROC_DIR / "sft_val.jsonl"
MODEL_NAME = "google/flan-t5-small"
ADAPTER_DIR = BASE_DIR / "models" / "flan_t5_small_lora"
TRANSLATOR_NAME = "Helsinki-NLP/opus-mt-tc-big-en-pt"

MAX_CTX_CHARS = 1200
TOP_K = 1

def load_jsonl(path, limit=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            ex = json.loads(line)
            data.append(ex)
    return data

def build_title_index(examples):
    titles = [ex["title"] for ex in examples]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(titles)
    return vec, X, titles

def retrieve_best(query_title, vec, X, examples, top_k=1):
    q = vec.transform([query_title])
    sims = cosine_similarity(q, X)[0]
    idx = sims.argsort()[::-1][:top_k]
    out = []
    for i in idx:
        ex = examples[i]
        out.append({
            "title": ex["title"],
            "uid": ex.get("uid", ""),
            "response": ex["response"],
            "score": float(sims[i]),
        })
    return out

def generate(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            length_penalty=1.0,
            early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def clean_generation(text: str) -> str:
    t = text.strip()
    t = t.replace("RESPOSTA", "").strip()
    t = re.sub(r'^\s*[\"“”]+', '', t)
    t = re.sub(r'[\"“”]+\s*$', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t

def jaccard_similarity(a: str, b: str) -> float:
    tok = lambda s: set(re.findall(r"\w+", s.lower()))
    A, B = tok(a), tok(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def looks_english(t: str) -> bool:
    low = " " + t.lower() + " "
    common = [" the ", " and ", " of ", " to ", " in ", " is ", " for ", " with ", " on ", " by ", " from "]
    ascii_ratio = sum(ch.isascii() for ch in t) / max(1, len(t))
    return any(w in low for w in common) and ascii_ratio > 0.98

def translate_to_pt(trans_tok, trans_model, text: str) -> str:
    batch = trans_tok([text], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = trans_model.generate(**batch, max_new_tokens=512)
    return trans_tok.decode(out[0], skip_special_tokens=True)

def main():
    print("Carregando tokenizer/modelo + adapter LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    print("Carregando tradutor pt-BR...")
    trans_tok = MarianTokenizer.from_pretrained(TRANSLATOR_NAME)
    trans_model = MarianMTModel.from_pretrained(TRANSLATOR_NAME)

    train = load_jsonl(TRAIN_PATH)
    val = load_jsonl(VAL_PATH)
    kb = train + val
    vec, X, _ = build_title_index(kb)

    tests = val[:5]

    for ex in tests:
        user_title = ex.get("title", "").strip()
        if not user_title:
            continue

        top = retrieve_best(user_title, vec, X, kb, top_k=TOP_K)[0]

        contexto = re.sub(r"\s+", " ", top["response"]).strip()[:MAX_CTX_CHARS]
        fonte_uid = top.get("uid", "")
        prompt = (
            "TAREFA: Use SOMENTE o CONTEXTO abaixo para responder em português do Brasil.\n"
            "Objetivo: reproduza fielmente a DESCRIÇÃO do produto em PT-BR.\n"
            "Regras: não faça perguntas, não repita o título, não acrescente comentários.\n"
            f"CONTEXTO (Fonte UID: {fonte_uid}): {contexto}\n\n"
            "RESPOSTA (apenas a descrição, em PT-BR):"
        )

        pred = generate(model, tokenizer, prompt)
        gen = clean_generation(pred)
        ctx_clean = re.sub(r"\s+", " ", top["response"]).strip()

        sim = jaccard_similarity(gen, ctx_clean)
        too_short = len(gen) < min(60, int(0.2 * len(ctx_clean)))
        bad_signals = ("?" in gen) or (len(gen.split()) <= 4)

        final = ctx_clean if (sim < 0.25 or too_short or bad_signals) else gen
        final_pt = translate_to_pt(trans_tok, trans_model, final) if looks_english(final) else final

        print("\n---")
        print("Título do usuário:", user_title)
        print("Fonte usada (uid):", fonte_uid)
        print("\nSAÍDA (PT-BR):\n", final_pt[:800], ("..." if len(final_pt) > 800 else ""))
        print("\nDescrição original (dataset):\n", ex["response"][:800], ("..." if len(ex["response"]) > 800 else ""))

if __name__ == "__main__":
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError(f"Arquivos processados não encontrados em {PROC_DIR}")
    main()

