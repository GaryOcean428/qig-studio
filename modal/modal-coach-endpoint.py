"""Qwen/Qwen3.5-4B on 1xL40S with SGLang — the developmental-coach LLM endpoint.

GPU right-sized L40S (was an H100 template): a 4B model (~8GB fp16) + DFLASH draft + KV cache fits well
inside L40S's 48GB; H100's 80GB is wasted spend for this size (Modal GPU-guide L40S recommendation). This
serves the coach's interpret/reframe (P10 witness role) via the OpenAI-compatible SGLang API, scale-to-zero
(min_containers=0, 5-min scaledown) so it costs nothing between coach calls, authenticated. The studio coach
(coach.py) reaches it through the OpenAI-compatible backend (QIG_COACH_ENDPOINT/-KEY) — no Ollama cloud, no
rate limit. Deploy: `modal deploy modal/modal-coach-endpoint.py`.

Serving metadata:
engine: sglang
base_model_repo_id: Qwen/Qwen3.5-4B
base_model_revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
model_family: qwen35

Deployed with MODAL_IMAGE_BUILDER_VERSION=2025.06"""
import modal


MINUTES = 60
DEFAULT_PORT = 8000
HF_IMAGE_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
}

MODEL_PATH = "/flash-endpoint-model/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SERVED_MODEL_NAME = "Qwen/Qwen3.5-4B"
ROUTING_REGION = "us-west"
REQUIRE_AUTHENTICATION = True
SPECULATIVE_DRAFT_MODEL_PATH = "/flash-endpoint-model/huggingface/hub/models--z-lab--Qwen3.5-4B-DFlash/snapshots/96899cc270945f554998309580b08a04a05a3187"
SGLANG_IMAGE_TAG = "modalresearch/sglang:nightly-dev-cu13-20260619-patched"
AUTOINFERENCE_UTILS_VERSION = "0.2.2"

GPU_TYPE = "L40S"
N_GPUS = 1
GPU = f"{GPU_TYPE}:{N_GPUS}"

SCALEDOWN_WINDOW = 5 * MINUTES
TARGET_INPUTS = 16
STARTUP_TIMEOUT = 60 * MINUTES

EXTRA_IMAGE_ENV = {
    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
}

serving_image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG)
    .uv_pip_install(
        f"autoinference-utils=={AUTOINFERENCE_UTILS_VERSION}",
    )
    .env(HF_IMAGE_ENV | EXTRA_IMAGE_ENV)
)

EXTRA_SERVER_ARGS = {
    "--mamba-scheduler-strategy": "extra_buffer",
    "--mamba-ssm-dtype": "float32",
    "--mem-fraction-static": "0.70",
    "--reasoning-parser": "qwen3",
    "--speculative-algorithm": "DFLASH",
    "--speculative-num-draft-tokens": "16",
    "--tool-call-parser": "qwen3_coder",
    "--trust-remote-code": "",
}

SERVER_ARGS = {
    "--served-model-name": SERVED_MODEL_NAME,
} | EXTRA_SERVER_ARGS

WARMUP_PAYLOAD = {
    "model": SERVED_MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with JSON facts about Tokyo."}],
    "max_tokens": 64,
    "temperature": 0,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "city_facts",
            "schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "population": {"type": "integer"},
                },
                "required": ["city", "population"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
}


app = modal.App(name="ep-qwen3-5-4b")


@app.server(
    image=serving_image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    min_containers=0,
    scaledown_window=SCALEDOWN_WINDOW,
    port=DEFAULT_PORT,
    routing_region=ROUTING_REGION,
    unauthenticated=not REQUIRE_AUTHENTICATION,
    exit_grace_period=25,
    startup_timeout=STARTUP_TIMEOUT,
    target_concurrency=TARGET_INPUTS,
    volumes={"/flash-endpoint-model": modal.Volume.from_name("endpoint-ep-5gwaDCdCkHLhV7Dvj79Cxd")},
)
class Server:
    @modal.enter()
    def startup(self):
        from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions

        self.endpoint = SGLangEndpoint(
            model_path=MODEL_PATH,
            worker_port=DEFAULT_PORT,
            tp=N_GPUS,
            speculative_model_path=SPECULATIVE_DRAFT_MODEL_PATH,
            extra_server_args=SERVER_ARGS,
            health_timeout=STARTUP_TIMEOUT,
            health_poll_interval=5.0,
        )
        self.endpoint.start()
        warmup_chat_completions(
            port=DEFAULT_PORT,
            payload=WARMUP_PAYLOAD,
            successful_requests=2,
            request_timeout=60.0,
        )
        print(f"{SERVED_MODEL_NAME} ({GPU}) sglang deployment is ready.")

    @modal.exit()
    def stop(self):
        if hasattr(self, "endpoint"):
            self.endpoint.stop()
