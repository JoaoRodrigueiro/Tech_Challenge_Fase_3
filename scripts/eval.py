import json, re, argparse, os, random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"
TRAIN_PATH = PROC_DIR / "sft_train.jsonl"
VAL_PATH   = PROC_DIR / "sft_val.jsonl"
MODEL_NAME = "google/flan-t5-small"
ADAPTER_DIR = BASE_DIR / "models" / "flan_t5_small_lora"

def load_jsonl(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def build_title_index(exs):
    titles = [e["title"] for e in exs]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(titles)
    return vec, X, titles

def retrieve_best(title, vec, X, titles, kb, top_k=1):
    q = vec.transform([title])
    sims = cosine_similarity(q, X)[0]
    idx = sims.argsort()[::-1][:top_k]
    return [kb[i] for i in idx]

def gen(model, tok, prompt, max_new_tokens=120):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            length_penalty=1.0,
            early_stopping=True,
        )
    return tok.decode(ids[0], skip_special_tokens=True)

def clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def rouge_l_f1(pred, ref):
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return sc.score(ref, pred)["rougeL"].fmeasure

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    args = ap.parse_args()

    print(">> Carregando dados...", flush=True)
    train = load_jsonl(TRAIN_PATH)
    val   = load_jsonl(VAL_PATH)
    random.seed(42)
    random.shuffle(val)
    val = val[:args.n]

    print(">> Construindo índice TF-IDF (usando apenas o treino)...", flush=True)
    vec, X, titles = build_title_index(train)

    print(">> Carregando modelo base...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print(">> Carregando modelo fine-tunado (LoRA)...", flush=True)
    ft = PeftModel.from_pretrained(base, ADAPTER_DIR)
    ft.eval()

    scores_base = []
    scores_ftctx = []

    for i, ex in enumerate(val, 1):
        title = ex["title"]
        ref   = clean(ex["response"])

        # baseline (sem contexto)
        p_base = (
            "Dado o TÍTULO de um produto da Amazon, responda apenas com a sua DESCRIÇÃO oficial.\n"
            f'Título: "{title}"\n'
            "Descrição:"
        )
        out_base = clean(gen(base, tok, p_base, max_new_tokens=args.max_new_tokens))
        s_base = rouge_l_f1(out_base, ref)
        scores_base.append(s_base)

        # FT + contexto (RAG simples por título no treino)
        best = retrieve_best(title, vec, X, titles, train, top_k=1)[0]
        ctx  = clean(best["response"])[:1200]
        p_ft = (
            "TAREFA: Use SOMENTE o CONTEXTO abaixo para responder em português do Brasil.\n"
            "Objetivo: reproduza fielmente a DESCRIÇÃO do produto em PT-BR.\n"
            "Regras: não faça perguntas, não repita o título, não acrescente comentários.\n"
            f"CONTEXTO: {ctx}\n\n"
            "RESPOSTA:"
        )
        out_ft = clean(gen(ft, tok, p_ft, max_new_tokens=args.max_new_tokens))
        s_ft = rouge_l_f1(out_ft, ref)
        scores_ftctx.append(s_ft)

        if i % 2 == 0 or i == args.n:
            print(f"[progresso] {i}/{args.n} | ROUGE-L base={s_base:.4f} | ft+ctx={s_ft:.4f}", flush=True)

    avg_base = sum(scores_base)/len(scores_base)
    avg_ft   = sum(scores_ftctx)/len(scores_ftctx)

    print("\n=== RESULTADOS ===", flush=True)
    print(f"Média ROUGE-L — baseline (sem contexto): {avg_base:.4f}", flush=True)
    print(f"Média ROUGE-L — FT + contexto: {avg_ft:.4f}", flush=True)

if __name__ == "__main__":
    main()