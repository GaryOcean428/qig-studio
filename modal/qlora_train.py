"""qig-studio's OWN Qwen-QLoRA Modal app — self-contained, OPTIONAL, NO vex dependency.

This is the qig-studio-native implementation of the qlora-train ASGI contract that
``targets/qwen_modal.py`` speaks. It is the OPTIONAL larger-Qwen weight-training path; the DEFAULT
Qwen path is local Ollama (``targets/qwen_local.py``). The whole app runs end-to-end WITHOUT this —
``QwenModalTarget`` is None-safe and only used when ``QIG_STUDIO_MODAL_URL`` points here.

NOTE on purity: Qwen is a Euclidean transformer (LANGUAGE loss regime). QLoRA fine-tunes it with
standard AdamW — that is CORRECT for a non-manifold LM, NOT a violation. Geometric purity (Fisher-Rao
/ natural gradient) governs the Δ⁶³ KERNEL targets, not Qwen. This file lives in ``modal/`` (outside
the ``src/`` geometric-purity scope) precisely because it trains a Euclidean model.

Deploy:  modal deploy modal/qlora_train.py   (sets QIG_STUDIO_MODAL_URL to the printed web URL)
Auth:    set a Modal secret ``qig-studio-modal`` with key ``api_key``; the client sends it as the
         ``x-api-key`` header only (never in the body).

Contract (matches targets/qwen_modal.py):
  GET  /health                                        → {"status": "ok"}
  POST /data-receive {filename, records:[{text, source}]}  → enqueue; {"records_written": n}
  POST /train {specialization, force}                 → spawn async QLoRA; 202 {"status": "accepted"}
  POST /infer {specialization, messages, max_tokens}  → {"text": ...}
  GET  /status {specialization}                       → {"state": "...", "step": n}
"""
from __future__ import annotations

import json
import os
import time

import modal

APP = modal.App("qig-studio-qlora")

# PyPI-pinned image (NEVER add_local_dir for published deps — image layers cache + reproducibility).
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "trl==0.12.1",
        "bitsandbytes==0.44.1",
        "accelerate==1.1.1",
        "datasets==3.1.0",
        "fastapi[standard]==0.115.5",
    )
)
VOL = modal.Volume.from_name("qig-studio-qlora", create_if_missing=True)
_DATA = "/data"
_BASE_MODEL = os.environ.get("QIG_STUDIO_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")


def _queue_path(spec: str) -> str:
    return f"{_DATA}/{spec}/curriculum.jsonl"


def _status_path(spec: str) -> str:
    return f"{_DATA}/{spec}/status.json"


def _adapter_dir(spec: str) -> str:
    return f"{_DATA}/{spec}/adapter"


# ───────────────────────────────── async QLoRA training ─────────────────────────────────

@APP.function(image=IMAGE, gpu="A10G", volumes={_DATA: VOL}, timeout=3600)
def train_qlora(specialization: str) -> dict:
    """Harvest the queued curriculum and run one QLoRA SFT pass, writing the adapter to the volume."""
    import pathlib

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    VOL.reload()
    queue = _queue_path(specialization)
    if not pathlib.Path(queue).exists():
        return {"state": "no-data", "step": 0}

    # source:"curriculum" → split the record into a prompt/completion chat (lm_loss is the signal).
    rows = []
    for line in pathlib.Path(queue).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        txt = rec.get("text", "")
        if rec.get("source") == "curriculum" and "\n" in txt:
            prompt, completion = txt.split("\n", 1)
        else:
            prompt, completion = "", txt
        rows.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": completion}]})
    if not rows:
        return {"state": "no-data", "step": 0}

    def _set_status(state, step):
        p = pathlib.Path(_status_path(specialization))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"state": state, "step": step, "ts": time.time()}), encoding="utf-8")
        VOL.commit()

    _set_status("training", 0)
    tok = AutoTokenizer.from_pretrained(_BASE_MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(_BASE_MODEL, quantization_config=bnb, device_map="auto")
    ds = Dataset.from_list(rows).map(
        lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)})
    # Euclidean LM → AdamW is correct (NOT a Δ⁶³ manifold object; see module docstring).
    cfg = SFTConfig(output_dir=f"/tmp/{specialization}", num_train_epochs=1, per_device_train_batch_size=1,
                    gradient_accumulation_steps=4, learning_rate=2e-4, optim="paged_adamw_8bit",
                    logging_steps=5, max_seq_length=1024, report_to=[])
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, peft_config=lora)
    trainer.train()
    trainer.save_model(_adapter_dir(specialization))
    _set_status("done", int(trainer.state.global_step))
    return {"state": "done", "step": int(trainer.state.global_step)}


# ───────────────────────────────── ASGI web contract ─────────────────────────────────

@APP.function(image=IMAGE, volumes={_DATA: VOL}, secrets=[modal.Secret.from_name("qig-studio-modal")])
@modal.asgi_app()
def web():
    import pathlib

    from fastapi import FastAPI, Header, HTTPException, Request

    api = FastAPI(title="qig-studio-qlora")
    _KEY = os.environ.get("api_key")

    def _auth(x_api_key: str | None):
        if _KEY and x_api_key != _KEY:
            raise HTTPException(status_code=401, detail="bad x-api-key")

    @api.get("/health")
    def health():
        return {"status": "ok", "app": "qig-studio-qlora"}

    @api.post("/data-receive")
    async def data_receive(req: Request, x_api_key: str | None = Header(default=None)):
        _auth(x_api_key)
        body = await req.json()
        spec = body.get("specialization", "genesis")
        records = body.get("records", [])
        VOL.reload()
        p = pathlib.Path(_queue_path(spec))
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        VOL.commit()
        return {"records_written": len(records), "success": True}

    @api.post("/train")
    async def train(req: Request, x_api_key: str | None = Header(default=None)):
        _auth(x_api_key)
        body = await req.json()
        spec = body.get("specialization", "genesis")
        train_qlora.spawn(spec)  # async — returns immediately; poll /status
        return {"status": "accepted", "specialization": spec, "success": True}

    @api.post("/infer")
    async def infer(req: Request, x_api_key: str | None = Header(default=None)):
        _auth(x_api_key)
        body = await req.json()
        spec = body.get("specialization", "genesis")
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 64))
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        VOL.reload()
        tok = AutoTokenizer.from_pretrained(_BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(_BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        adir = _adapter_dir(spec)
        if pathlib.Path(adir).exists():
            model = PeftModel.from_pretrained(model, adir)
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=max_tokens)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return {"text": text, "success": True}

    @api.get("/status")
    def status(specialization: str = "genesis"):
        VOL.reload()
        p = pathlib.Path(_status_path(specialization))
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"state": "idle", "step": 0}

    return api
