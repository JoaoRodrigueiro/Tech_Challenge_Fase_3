import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

BASE = Path(__file__).resolve().parents[1]
PROC = BASE/"data"/"processed"
TRAIN, VAL = PROC/"sft_train.jsonl", PROC/"sft_val.jsonl"
OUT = BASE/"data"/"dense_index"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "intfloat/multilingual-e5-small"

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def main():
    kb = load_jsonl(TRAIN) + load_jsonl(VAL)
    titles = [x["title"] for x in kb]
    model = SentenceTransformer(MODEL)
    embs = model.encode([f"passage: {t}" for t in titles], normalize_embeddings=True, batch_size=128, convert_to_numpy=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    np.save(OUT/"title_embeds.npy", embs)
    with open(OUT/"meta.jsonl", "w", encoding="utf-8") as f:
        for ex in kb:
            f.write(json.dumps({"title": ex["title"], "uid": ex.get("uid",""), "response": ex["response"]}, ensure_ascii=False)+"\n")
    faiss.write_index(index, str(OUT/"faiss.index"))
    print("ok:", embs.shape, "salvo em", OUT)

if __name__ == "__main__":
    main()
