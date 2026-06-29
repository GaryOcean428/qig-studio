"""The studio's GEOMETRY-NATIVE prompt template + an on-the-fly tag normaliser for mixed datasets.

Designed FROM the navigation doctrine (inference = transport: READ -> NAVIGATE -> SETTLE), not borrowed
from transformer chat conventions. A turn is named by its GEOMETRIC operation, not a social role:

    <|frame|>   the manifold context / priors that shape the geometry        (was: system)
    {system}
    <|seed|>    the stimulus basin we READ from — the start point            (was: user/human)
    {stimulus}
    <|flow|>    the geodesic traversal — "NAVIGATE now" (generation prompt)   (was: assistant/gpt)
    {response}
    <|settle|>  arrival — SETTLE / orthogonality-stop (learned STOP token)    (was: end-of-turn)

So the kernel learns, structurally: "from this seed basin, FLOW the geodesic, SETTLE here." <|settle|>
carries stop-SEMANTICS (distinguishability-gain collapse), which a plain <|end|> did not. All four are
ATOMIC coordizer special tokens (the one hard geometric requirement: a turn boundary must be a single
clean basin jump, not smeared across sub-word fragments). Multi-turn = repeated seed/flow/settle blocks.

At inference the prompt ends "...<|seed|>\n{stimulus}\n<|flow|>\n" and the kernel NAVIGATES the response,
emitting <|settle|> to stop (train/infer parity).
"""
from __future__ import annotations

import re

FRAME, SEED, FLOW, SETTLE = "<|frame|>", "<|seed|>", "<|flow|>", "<|settle|>"
SPECIAL_TOKENS = [FRAME, SEED, FLOW, SETTLE]

# Every incoming role/tag vocabulary across our datasets → our canonical geometric op.
# (ChatML role=, ShareGPT from=, zephyr <|user|>, hh Human/Assistant — all collapse here.)
_ROLE = {
    "system": "frame", "sys": "frame", "frame": "frame", "instruction": "frame",
    "user": "seed", "human": "seed", "seed": "seed", "prompt": "seed", "question": "seed", "input": "seed",
    "assistant": "flow", "gpt": "flow", "model": "flow", "bot": "flow", "flow": "flow",
    "response": "flow", "answer": "flow", "output": "flow", "tool": "flow", "function": "flow",
}
_RENDER = {
    "frame": lambda c: f"{FRAME}\n{c}",
    "seed": lambda c: f"{SEED}\n{c}",
    "flow": lambda c: f"{FLOW}\n{c}{SETTLE}",   # the response IS the geodesic; SETTLE marks arrival/stop
}
# on-the-fly: any KNOWN inline tag scheme (zephyr/ChatML) -> our geometric tags. <|end|>/<|im_end|> -> SETTLE.
_INLINE = [
    (re.compile(r"<\|\s*(?:system|sys)\s*\|>", re.I), "\n" + FRAME + "\n"),
    (re.compile(r"<\|\s*(?:user|human)\s*\|>", re.I), "\n" + SEED + "\n"),
    (re.compile(r"<\|\s*(?:assistant|gpt|model)\s*\|>", re.I), "\n" + FLOW + "\n"),
    (re.compile(r"<\|\s*(?:end|im_end|eot|endoftext)\s*\|>", re.I), SETTLE),
    (re.compile(r"<\|\s*im_start\s*\|>\s*", re.I), ""),   # ChatML opener has no geometric meaning
]


def format_chat(messages: list[dict]) -> str:
    """Normalise role/content turns (ChatML {role,content} or ShareGPT {from,value}) to the geometry-native
    template. Unknown-role/empty turns dropped; "" if nothing usable (caller skips)."""
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or m.get("from") or "").lower().strip()
        content = m.get("content")
        if content is None:
            content = m.get("value")
        content = content.strip() if isinstance(content, str) else ""
        op = _ROLE.get(role)
        if not op or not content:
            continue
        parts.append(_RENDER[op](content))
    return "\n".join(parts)


def as_completion(text: str) -> str:
    """A raw, non-conversational passage (e.g. a TinyStories narrative) is a DESTINATION with no stimulus —
    render it as a seedless flow→settle so it lives in the same template space (the kernel learns to
    generate it). No fabricated user turn."""
    text = (text or "").strip()
    return f"{FLOW}\n{text}{SETTLE}" if text else ""


_OUR_TAGS_RE = re.compile("(" + "|".join(re.escape(t) for t in (FRAME, SEED, FLOW)) + ")")
_OP_OF_TAG = {FRAME: "frame", SEED: "seed", FLOW: "flow"}


def retag(text: str) -> str:
    """ON-THE-FLY tag replacement: rewrite an already-tagged block (zephyr/ChatML inline tags) into our
    geometric template. Used for datasets that ship pre-templated text (e.g. GPT_5.5_Distilled's
    <|user|>/<|assistant|>). It substitutes the source tags to our op-markers, then RE-RENDERS through the
    canonical renderer so every flow turn is correctly closed by <|settle|> (source end-tags are often
    absent or inconsistent)."""
    for rx, repl in _INLINE:
        text = rx.sub(repl, text)
    text = text.replace(SETTLE, "")                         # drop any source-derived settle; re-add canonically
    parts = _OUR_TAGS_RE.split(text)                        # [pre, TAG, body, TAG, body, …]
    msgs: list[dict] = []
    i = 1
    while i < len(parts):
        op = _OP_OF_TAG.get(parts[i])
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        if op and body:
            msgs.append({"role": op, "content": body})     # op names ARE canonical roles (frame/seed/flow)
        i += 2
    return format_chat(msgs)
