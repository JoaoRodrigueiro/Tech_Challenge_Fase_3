import json
from pathlib import Path
import re
import numpy as np
import faiss
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"
TRAIN_PATH = PROC_DIR / "sft_train.jsonl"
VAL_PATH = PROC_DIR / "sft_val.jsonl"
MODEL_NAME = "google/flan-t5-small"
ADAPTER_DIR = BASE_DIR / "models" / "flan_t5_small_lora"
TRANSLATOR_NAME = "Helsinki-NLP/opus-mt-tc-big-en-pt"

MAX_CTX_CHARS = 1200
TOP_K = 3
SIM_THRESHOLD = 0.25


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


def translate_to_pt(trans_tok, trans_model, text: str) -> str:
    batch = trans_tok([text], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = trans_model.generate(**batch, max_new_tokens=512)
    return trans_tok.decode(out[0], skip_special_tokens=True)


def build_prompt(contexto, uid, trans_tok, trans_model):
    contexto_pt = translate_to_pt(trans_tok, trans_model, contexto)
    contexto_pt = re.sub(r"\s+", " ", contexto_pt).strip()[:MAX_CTX_CHARS]
    return (
        "TAREFA: Use SOMENTE o CONTEXTO abaixo para responder em português do Brasil.\n"
        "Objetivo: reproduza fielmente a DESCRIÇÃO do produto em PT-BR.\n"
        "Regras: não faça perguntas, não repita o título, não acrescente comentários.\n"
        f"CONTEXTO (Fonte UID: {uid}): {contexto_pt}\n\n"
        "RESPOSTA (apenas a descrição, em PT-BR):"
    )


def dense_retrieve(query, dense_model, dense_index, meta, k=3):
    q = dense_model.encode([f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True)
    D, I = dense_index.search(q, k)
    out = []
    for idx, score in zip(I[0], D[0]):
        ex = meta[idx]
        out.append({"title": ex["title"], "uid": ex.get("uid",""), "response": ex["response"], "score": float(score)})
    return out


def answer_with_context(model, tokenizer, trans_tok, trans_model, dense_model, dense_index, meta, query):
    cands = dense_retrieve(query, dense_model, dense_index, meta, k=TOP_K)
    for top in cands:
        prompt = build_prompt(top["response"], top["uid"], trans_tok, trans_model)
        pred = generate(model, tokenizer, prompt)
        gen = clean_generation(pred)
        ctx_clean = re.sub(r"\s+", " ", top["response"]).strip()
        sim = jaccard_similarity(gen, ctx_clean)
        too_short = len(gen) < min(60, int(0.2 * len(ctx_clean)))
        bad_signals = ("?" in gen) or (len(gen.split()) <= 4)
        final = ctx_clean if (sim < SIM_THRESHOLD or too_short or bad_signals) else gen
        final_pt = translate_to_pt(trans_tok, trans_model, final)
        if final_pt:
            return final_pt, top["uid"], top["title"], cands
    top = cands[0]
    return translate_to_pt(trans_tok, trans_model, top["response"]), top["uid"], top["title"], cands


def main():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError(f"Arquivos processados não encontrados em {PROC_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    trans_tok = MarianTokenizer.from_pretrained(TRANSLATOR_NAME)
    trans_model = MarianMTModel.from_pretrained(TRANSLATOR_NAME)
    
    DENSE_DIR = BASE_DIR / "data" / "full_index"
    if not (DENSE_DIR / "faiss.index").exists():
        raise FileNotFoundError("Rode antes: python scripts/build_dense_index_full.py")

    dense_model = SentenceTransformer("intfloat/multilingual-e5-small")
    dense_index = faiss.read_index(str(DENSE_DIR / "faiss.index"))
    meta = [json.loads(l) for l in open(DENSE_DIR / "meta.jsonl", "r", encoding="utf-8")]

    print("Pronto. Digite um TÍTULO de produto (ou 'sair').")
    while True:
        try:
            query = input("\nTítulo: ").strip()
            if not query:
                continue
            if query.lower() in {"sair", "exit", "quit"}:
                break
            answer, uid, matched_title, evidences = answer_with_context(
                model, tokenizer, trans_tok, trans_model, dense_model, dense_index, meta, query
            )
            print("\nDescrição (PT-BR):")
            print(answer)
            print(f"\nFonte UID: {uid}")
            print(f"Título encontrado: {matched_title}")
            print("\nEvidências (top-k):")
            for c in evidences:
                print(f"- UID {c['uid']} | score={c['score']:.3f} | título: {c['title'][:80]}")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
