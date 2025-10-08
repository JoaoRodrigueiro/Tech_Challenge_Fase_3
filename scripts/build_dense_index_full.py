import json, gzip, re, argparse
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
GZ = DATA / "trn.json.gz"
OUT = DATA / "full_index"

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def count_lines(p: Path) -> int:
    if not p.exists(): return 0
    c = 0
    with open(p, "r", encoding="utf-8") as f:
        for _ in f: c += 1
    return c

def stream_gz(skip: int, limit: int | None):
    with gzip.open(GZ, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip: continue
            yield json.loads(line)
            if limit is not None and (i - skip + 1) >= limit: break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=100_000)
    args = parser.parse_args()

    if not GZ.exists():
        raise FileNotFoundError(f"Não encontrei {GZ}")
    OUT.mkdir(parents=True, exist_ok=True)

    meta_path = OUT / "meta.jsonl"
    index_path = OUT / "faiss.index"

    done = count_lines(meta_path)
    if index_path.exists() and done > 0:
        index = faiss.read_index(str(index_path))
        dim = index.d
        meta_f = open(meta_path, "a", encoding="utf-8")
    else:
        index = None
        dim = None
        meta_f = open(meta_path, "w", encoding="utf-8")

    model = SentenceTransformer(args.model, device="cpu")

    buf_titles, buf_meta = [], []
    processed = done
    pbar = tqdm(total=args.max if args.max is not None else None, unit="it", initial=0, desc="Indexando")

    for ex in stream_gz(skip=done, limit=args.max):
        title = clean(ex.get("title", ""))
        uid = ex.get("uid", "")
        content = clean(ex.get("content", ""))
        if not title or not content: 
            continue
        buf_titles.append(f"passage: {title}")
        buf_meta.append({"title": title, "uid": uid, "response": content})

        if len(buf_titles) >= args.batch:
            embs = model.encode(
                buf_titles, normalize_embeddings=True,
                convert_to_numpy=True, batch_size=args.batch, show_progress_bar=False
            ).astype(np.float32)

            if index is None:
                dim = embs.shape[1]
                index = faiss.IndexFlatIP(dim)

            index.add(embs)
            for m in buf_meta:
                meta_f.write(json.dumps(m, ensure_ascii=False) + "\n")
            meta_f.flush()

            processed += len(buf_titles)
            pbar.update(len(buf_titles))
            buf_titles.clear(); buf_meta.clear()

            if processed % args.save_every == 0:
                faiss.write_index(index, str(index_path))

    if buf_titles:
        embs = model.encode(
            buf_titles, normalize_embeddings=True,
            convert_to_numpy=True, batch_size=args.batch, show_progress_bar=False
        ).astype(np.float32)
        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
        index.add(embs)
        for m in buf_meta:
            meta_f.write(json.dumps(m, ensure_ascii=False) + "\n")
        meta_f.flush()
        processed += len(buf_titles)
        pbar.update(len(buf_titles))

    faiss.write_index(index, str(index_path))
    meta_f.close()
    pbar.close()
    print(f"OK. Total indexados agora: ~{processed} itens em {OUT}")

if __name__ == "__main__":
    main()
