#!/usr/bin/env python3
"""
run_mixtral.py

Loads a 4-bit Mixtral-8x7B-Instruct model on GPU and exposes a simple
`ask()` function. Run it from the command line or import into a service.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download, login as hf_login

# 1️⃣ Setup paths and cache on high-speed local scratch
SCRATCH = "/scratch/sjulaka7"
SCRATCH_CACHE = os.path.join(SCRATCH, "hf_cache")
os.makedirs(SCRATCH_CACHE, exist_ok=True)
os.environ["HF_HOME"] = SCRATCH_CACHE

# 1️⃣a Authenticate to HF (try env var first, then hardcoded token)
hf_token = os.getenv("HF_HUB_TOKEN", "hgGcpZuPDoxIMFZhkEsnR")
if hf_token:
    try:
        hf_login(token=hf_token)
        print(f" Successfully authenticated with Hugging Face")
    except Exception as e:
        print(f" HF authentication failed: {e}")
        print("Proceeding without authentication...")

# 2️⃣ Quantization config: 4-bit NF4 + double quant
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 3️⃣ Model identifier
MODEL_ID = "mistralai/mixtral-8x7b-instruct-v0.1"

# 4️⃣ Pre-download the model once (optional but speeds up subsequent runs)
def ensure_model_cached(model_id: str):
    print(f"📥 Ensuring {model_id} is cached in {SCRATCH_CACHE}...")
    snapshot_download(
        repo_id=model_id,
        cache_dir=SCRATCH_CACHE,
        token=hf_token
    )

# 5️⃣ Load model & tokenizer
def load_model(model_id: str = MODEL_ID):
    """
    Load tokenizer & 4-bit quantized Mixtral-8x7B-Instruct model onto GPU.
    """
    print(f"🔽 Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    print(f"🔽 Loading Mixtral model (4-bit) on GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).eval()

    dev = next(model.parameters()).device
    print(f"✅ Model loaded on {dev}")
    return tokenizer, model

# 6️⃣ Generation helper
def ask(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.92
) -> str:
    """
    Generate a response from the model given a text prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# 7️⃣ CLI entrypoint
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Mixtral 8×7B-Instruct (4-bit) inference on GPU."
    )
    parser.add_argument(
        "--prompt", "-p", type=str,
        help="Text prompt to send to the model",
        required=True
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max new tokens to generate"
    )
    parser.add_argument(
        "--temp", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.92,
        help="Nucleus sampling top_p"
    )
    args = parser.parse_args()

    # Ensure model is cached
    ensure_model_cached(MODEL_ID)

    # Load model & tokenizer
    tokenizer, model = load_model(MODEL_ID)

    # Generate and print
    response = ask(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p
    )
    print("\n─── Response ───")
    print(response)

if __name__ == "__main__":
    main()
