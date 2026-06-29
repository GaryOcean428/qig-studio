"""One-shot Modal QLoRA: fine-tune Qwen3.5-4B on the claude-fable-5 agent traces, on an A10G.

Why Modal: the local 4 GB GTX 1650 Ti cannot train a 4B model (4-bit base ~3 GB + training prep ~2.4 GB
> 3.6 GB → CUDA OOM). An A10G (24 GB, ~$1.10/hr) holds 4B QLoRA comfortably and is far cheaper than A100.

Optimizer note (purity): Qwen has EUCLIDEAN internals (dot-product attention, additive residual stream), so
AdamW is the correct optimizer for its weights — exactly the studio Modal app's stated doctrine. The QIG
Fisher-Rao / natural-gradient optimizer governs the Δ⁶³ KERNELS (the separate kernel-retrain step), NOT
Qwen. So this uses paged_adamw_8bit; that is the principled choice here, not a shortcut.

Run:
  cd qig-studio && uv run --project . modal run modal/qwen_qlora_fable.py
Download the adapter when done:
  uv run --project . modal volume get qig-qwen-fable-adapter /fable-adapter ./runs/qwen_fable_adapter
"""
import json
import os

import modal

APP = modal.App("qig-qwen-qlora-fable")

_FABLE = os.path.join(os.path.dirname(__file__), "..", "runs", "fable_qlora.jsonl")
_BASE = os.environ.get("QIG_QWEN_BASE", "Qwen/Qwen3.5-4B")  # match the boundary-peer model

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install(
        "torch",
        "torchvision",       # the Qwen3.5 AutoTokenizer pulls in a VL image processor that imports torchvision+PIL
        "pillow",
        "transformers>=4.44",
        "peft>=0.12",
        "bitsandbytes>=0.43",
        "datasets>=2.18",
        "accelerate>=0.30",
        "sentencepiece",     # some Qwen tokenizer paths need it
        # NOTE: NO trl — trl 1.7's SFTTrainer mis-detects Qwen3.5 as a VLM and crashes
        # (text_config / chunked-CE-lm-head). Use plain transformers.Trainer instead (qig-applied's approach).
    )
    .add_local_file(_FABLE, "/data/fable.jsonl")
)
VOL = modal.Volume.from_name("qig-qwen-fable-adapter", create_if_missing=True)


@APP.function(image=IMAGE, gpu="A10G", timeout=3600, volumes={"/out": VOL},
              secrets=[modal.Secret.from_name("custom-secret")])  # HF_TOKEN → authenticated, fast model download
def train() -> dict:
    import pathlib

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments)

    rows = [json.loads(ln) for ln in pathlib.Path("/data/fable.jsonl").read_text().splitlines() if ln.strip()]
    print(f"[qlora] {len(rows)} fable conversations | base {_BASE}", flush=True)

    tok = AutoTokenizer.from_pretrained(_BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    MAXLEN = 2048

    # COMPLETION-ONLY masking (manual — Qwen3.5's template has no {% generation %} markers, so the built-in
    # assistant-token mask is unusable). Prefix-diff: for each assistant turn, the tokens between the render
    # of messages[:i] and messages[:i+1] are that turn (header + content + <|im_end|>). Train labels ONLY on
    # those (incl. <|im_end|> → learns to STOP); mask user/system turns to -100 so it never learns to GENERATE
    # the user side or continue the whole conversation (the stray-extra-turn artifact). enable_thinking=False
    # keeps the format consistent with the plain (no-<think>) fable narration.
    def _ct(msgs):
        # return_dict=False → a flat token-id LIST (tokenize=True alone returns a BatchEncoding whose len()
        # is the #keys, which silently zeroed every span — the 0-trainable-tokens bug).
        return tok.apply_chat_template(msgs, tokenize=True, enable_thinking=False, return_dict=False)

    def _build(messages):
        ids = _ct(messages)
        labels = [-100] * len(ids)
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            a = len(_ct(messages[:i])) if i else 0
            b = len(_ct(messages[:i + 1]))
            for j in range(a, min(b, len(ids))):
                labels[j] = ids[j]
        return ids[:MAXLEN], labels[:MAXLEN]

    examples = []
    for r in rows:
        ids, labels = _build(r["messages"])
        if any(lbl != -100 for lbl in labels):      # keep only convs with at least one trainable assistant token
            examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    ds = Dataset.from_list(examples)
    print(f"[qlora] completion-only: {len(ds)} examples | "
          f"trainable tokens {sum(sum(1 for x in e['labels'] if x!=-100) for e in examples):,}", flush=True)

    def _collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        def _p(x, v): return x + [v] * (m - len(x))
        return {"input_ids": torch.tensor([_p(b["input_ids"], pad) for b in batch]),
                "attention_mask": torch.tensor([_p(b["attention_mask"], 0) for b in batch]),
                "labels": torch.tensor([_p(b["labels"], -100) for b in batch])}

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(_BASE, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)

    # plain transformers.Trainer (NO trl) — avoids trl's Qwen3.5 VLM-misdetection crash
    args = TrainingArguments(output_dir="/tmp/out", num_train_epochs=5, per_device_train_batch_size=1,
                             gradient_accumulation_steps=8, learning_rate=1e-4, warmup_steps=10,
                             logging_steps=2, bf16=True, optim="paged_adamw_8bit", report_to=[], save_strategy="no")
    trainer = Trainer(model=model, train_dataset=ds, args=args, data_collator=_collate)
    trainer.train()
    model.save_pretrained("/out/fable-adapter")
    tok.save_pretrained("/out/fable-adapter")
    VOL.commit()
    hist = [h for h in trainer.state.log_history if "loss" in h]
    return {"steps": int(trainer.state.global_step), "convs": len(rows),
            "first_loss": hist[0]["loss"] if hist else None, "last_loss": hist[-1]["loss"] if hist else None,
            "base": _BASE}


@APP.local_entrypoint()
def main() -> None:
    print("RESULT:", train.remote())
    print("Download: modal volume get qig-qwen-fable-adapter /fable-adapter ./runs/qwen_fable_adapter")
