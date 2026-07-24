"""Qwen3.5-4B developmental-coach endpoint — self-contained vLLM OpenAI server, scale-to-zero.

Replaces modal-coach-endpoint.py, whose SGLang template pinned a pre-staged model Volume
(endpoint-ep-...) that does not exist → deploy failed. This one is self-contained: vLLM downloads
Qwen/Qwen3.5-4B from HF into an AUTO-CREATED cache Volume (create_if_missing — no manual staging), cached
across cold starts. Serves the OpenAI-compatible /v1/chat/completions the coach's OpenAICompatLLM calls.

Cost (per the endpoints pricing guide): min_containers=0 → $0 while idle; L40S right-sized for a 4B
(~8GB weights fit with room for KV; H100 would be wasted spend); billed per-second only while a coach call
is in flight, then scales to zero after `scaledown_window`. The coach reaches it via QIG_COACH_ENDPOINT
(the deployed URL) + QIG_COACH_KEY (Bearer, = the vllm --api-key from the qig-coach-key secret).

Deploy: modal deploy modal/modal-coach-vllm.py
"""
import os
import socket
import subprocess

import modal

MINUTES = 60
VLLM_PORT = 8000
MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"   # pinned (matches the coordizer-era coach model)
SERVED_NAME = "qwen3.5:4b"                                    # the coach requests this exact name

app = modal.App("qig-coach-qwen35-4b")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    # vLLM 0.25.1 (latest) bundles transformers 5.x — Qwen3.5's `qwen3_5` arch needs transformers>=4.57
    # (the model config's transformers_version); vLLM 0.13.0 was too old and exited on the unknown arch.
    .uv_pip_install("vllm==0.25.1", "huggingface-hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# AUTO-CREATED caches (NOT pre-staged): weights + vLLM compile artifacts persist across cold starts.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def _wait_ready(proc: "subprocess.Popen") -> None:
    while True:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {proc.returncode}")


@app.cls(
    image=vllm_image,
    gpu="L40S",                          # right-sized for a 4B (H100 = wasted spend)
    scaledown_window=5 * MINUTES,        # scale to zero 5 min after the last coach call → $0 idle
    timeout=10 * MINUTES,                # container-start budget (first cold start downloads the weights)
    min_containers=0,                    # SCALE-TO-ZERO — the cost guard
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("qig-coach-key")],
)
@modal.concurrent(max_inputs=16)
class CoachServer:
    @modal.enter()
    def start(self):
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--host", "0.0.0.0", "--port", str(VLLM_PORT),
            "--served-model-name", SERVED_NAME, "llm",
            "--api-key", os.environ["QIG_COACH_KEY"],       # Bearer auth (matches the coach client)
            "--max-model-len", "8192",
        ]
        self.proc = subprocess.Popen(cmd)
        _wait_ready(self.proc)

    @modal.exit()
    def stop(self):
        if getattr(self, "proc", None) is not None:
            self.proc.terminate()

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass
