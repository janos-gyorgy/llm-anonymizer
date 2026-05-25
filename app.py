"""
llm-anonymizer — privacy-preserving text transformer.

POST /anonymize   → replaces sensitive entities with tokens, returns mapping
POST /deanonymize → restores tokens to original values (pure string substitution)

The anonymization step uses a local LLM (llama.cpp) to extract entities.
Deanonymization needs no LLM — it's a deterministic string replacement.
"""

import os
import json
import logging
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="llm-anonymizer")

LLAMA_BASE = os.environ["LLAMA_BASE_URL"]

ANONYMIZE_PROMPT = """You are a data anonymization engine. Extract all sensitive entities from the text and replace them with placeholder tokens.

Entity types to detect and their token prefixes:
- People's names → [PERSON_N]
- Company / organization names → [ORG_N]
- Project / product names → [PROJECT_N]
- Email addresses → [EMAIL_N]
- IP addresses → [IP_N]
- Hostnames / URLs (excluding generic public ones like github.com) → [HOST_N]
- Credentials, tokens, API keys → [CRED_N]
- Phone numbers → [PHONE_N]
- Physical addresses → [ADDR_N]

Rules:
- N starts at 1 for each type and increments per unique value
- Use the same token for repeated occurrences of the same entity
- Keep all other text exactly as-is
- Respond ONLY with valid JSON, no explanation, no markdown

Output format:
{"anonymized": "<text with tokens>", "mapping": {"[PERSON_1]": "John Smith", ...}}

Text to anonymize:
"""


class AnonymizeRequest(BaseModel):
    text: str


class AnonymizeResponse(BaseModel):
    anonymized: str
    mapping: dict[str, str]


class DeanonymizeRequest(BaseModel):
    text: str
    mapping: dict[str, str]


class DeanonymizeResponse(BaseModel):
    text: str


@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize(req: AnonymizeRequest):
    if not req.text.strip():
        return AnonymizeResponse(anonymized=req.text, mapping={})

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{LLAMA_BASE}/chat/completions",
            json={
                "model": "local",
                "messages": [
                    {"role": "user", "content": ANONYMIZE_PROMPT + req.text[:4000]},
                ],
                "max_tokens": 1024,
                "temperature": 0,
                "stream": False,
            },
            headers={"Authorization": "Bearer sk-no-key"},
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    log.info("raw anonymizer output: %s", raw[:200])

    # Strip markdown code fences if the model wraps its output
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
        anonymized = parsed["anonymized"]
        mapping = parsed.get("mapping", {})
    except (json.JSONDecodeError, KeyError) as exc:
        log.error("failed to parse anonymizer output: %s — %s", exc, raw[:300])
        raise HTTPException(status_code=502, detail=f"Anonymizer returned unparseable output: {exc}")

    log.info("anonymized %d entities", len(mapping))
    return AnonymizeResponse(anonymized=anonymized, mapping=mapping)


@app.post("/deanonymize", response_model=DeanonymizeResponse)
async def deanonymize(req: DeanonymizeRequest):
    text = req.text
    # Sort by token length descending to avoid partial replacements
    for token, original in sorted(req.mapping.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(token, original)
    return DeanonymizeResponse(text=text)


@app.get("/health")
async def health():
    return {"status": "ok"}
