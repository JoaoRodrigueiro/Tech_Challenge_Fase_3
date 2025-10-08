import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = Path(__file__).resolve().parents[1]
VAL_PATH = BASE_DIR / "data" / "processed" / "sft_val.jsonl"
OUT_PATH = BASE_DIR / "data" / "processed" / "base_model_predictions.jsonl"

MODEL_NAME = "google/flan-t5-small"  

def load_samples(n=5):
    samples = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            samples.append(ex)
    return samples

def main():
    print("Carregando modelo/base...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    samples = load_samples(5)
    preds = []

    for ex in samples:
        prompt = ex["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,      
            num_beams=1
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        preds.append({
            "prompt": prompt,
            "expected_response": ex["response"],
            "base_model_output": text
        })
        print("\n---")
        print("PROMPT:\n", prompt)
        print("\nSAÍDA (modelo base):\n", text)
        print("\nESPERADO (descrição real):\n", ex["response"][:300], ("..." if len(ex["response"])>300 else ""))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n✅ Predições salvas em: {OUT_PATH}")

if __name__ == "__main__":
    main()
