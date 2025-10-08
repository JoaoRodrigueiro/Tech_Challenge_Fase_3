import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parents[1]
VAL_PATH = BASE_DIR / "data" / "processed" / "sft_val.jsonl"
OUT_PATH = BASE_DIR / "data" / "processed" / "finetuned_predictions.jsonl"
MODEL_NAME = "google/flan-t5-small"
ADAPTER_DIR = BASE_DIR / "models" / "flan_t5_small_lora"

def load_samples(n=5):
    samples = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line))
    return samples

def generate(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             
            num_beams=4,                  
            no_repeat_ngram_size=4,       
            repetition_penalty=1.15,      
            length_penalty=1.0,           
            early_stopping=True           
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    print("Carregando tokenizer e modelo base...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print(f"Carregando adapter LoRA de: {ADAPTER_DIR}")
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    ft_model.eval()

    samples = load_samples(5)
    preds = []

    for ex in samples:
        prompt = ex["prompt"]
        expected = ex["response"]

        pred = generate(ft_model, tokenizer, prompt)

        preds.append({
            "prompt": prompt,
            "expected_response": expected,
            "finetuned_output": pred
        })

        print("\n---")
        print("PROMPT:\n", prompt)
        print("\nSAÍDA (modelo fine-tunado):\n", pred[:600], ("..." if len(pred) > 600 else ""))
        print("\nESPERADO (descrição real):\n", expected[:600], ("..." if len(expected) > 600 else ""))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n✅ Predições (fine-tunado) salvas em: {OUT_PATH}")

if __name__ == "__main__":
    main()
