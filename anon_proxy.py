"""
anon-proxy — OpenAI-compatible chat proxy with PII anonymization.

Sits between Open WebUI and NVIDIA NIM.
Anonymizes all user messages before sending, deanonymizes the response.
Exposes a single model ID so Open WebUI shows one clean option.
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="anon-proxy")

UPSTREAM_BASE  = os.environ["UPSTREAM_BASE_URL"]
UPSTREAM_KEY   = os.environ["UPSTREAM_API_KEY"]
UPSTREAM_MODEL = os.environ.get("UPSTREAM_MODEL", "meta/llama-3.3-70b-instruct")
ANONYMIZER_URL = os.environ["ANONYMIZER_BASE_URL"]  # http://llm-anonymizer.web-ai-engine.svc:8000

MODEL_ID = "anon-nim-70b"


@app.get("/v1/models")
async def models():
    return JSONResponse({"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "created": 1, "owned_by": "anon-proxy"}
    ]})


async def anonymize(text: str) -> tuple[str, dict]:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{ANONYMIZER_URL}/anonymize", json={"text": text})
        r.raise_for_status()
    data = r.json()
    return data["anonymized"], data["mapping"]


async def deanonymize(text: str, mapping: dict) -> str:
    if not mapping:
        return text
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{ANONYMIZER_URL}/deanonymize", json={"text": text, "mapping": mapping})
        r.raise_for_status()
    return r.json()["text"]


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    body["model"] = UPSTREAM_MODEL
    body.pop("stream", None)  # anon-proxy doesn't support streaming (deanonymize needs full response)

    messages = body.get("messages", [])

    # Collect all user text in one batch for consistent entity mapping across turns
    user_texts = [
        (i, msg["content"])
        for i, msg in enumerate(messages)
        if msg.get("role") == "user" and isinstance(msg.get("content"), str) and msg["content"].strip()
    ]

    mapping: dict[str, str] = {}
    if user_texts:
        combined = "\n---\n".join(text for _, text in user_texts)
        _, mapping = await anonymize(combined)
        log.info("mapping has %d entity pairs", len(mapping))

        # Apply the shared mapping to each user message via deanonymize-in-reverse
        for i, text in user_texts:
            anon_text = text
            for original, replacement in sorted(mapping.items(), key=lambda x: len(x[1]), reverse=True):
                anon_text = anon_text.replace(original, replacement)
            messages[i] = {**messages[i], "content": anon_text}

    body["messages"] = messages
    headers = {"Authorization": f"Bearer {UPSTREAM_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{UPSTREAM_BASE}/chat/completions", json=body, headers=headers)

    data = resp.json()
    if mapping and "choices" in data:
        for choice in data["choices"]:
            msg = choice.get("message", {})
            if "content" in msg and msg["content"]:
                msg["content"] = await deanonymize(msg["content"], mapping)

    return JSONResponse(content=data, status_code=resp.status_code)


@app.get("/health")
async def health():
    return {"status": "ok"}
