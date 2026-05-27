"""
llm-anonymizer — privacy-preserving text transformer.

POST /anonymize   → replaces PII with semantically similar fakes, returns mapping
POST /deanonymize → restores originals via string substitution (no LLM needed)

Uses Ollama running Anonymizer-1.7B (eternisai, Qwen3-based tool-calling fine-tune).
The model outputs a replace_entities tool call with {original, replacement} pairs.
"""

import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="llm-anonymizer")

OLLAMA_BASE = os.environ["OLLAMA_BASE_URL"]
MODEL = os.environ.get("ANONYMIZER_MODEL", "hf.co/gabriellarson/Anonymizer-1.7B-GGUF")

SYSTEM_PROMPT = """You are an anonymizer. Identify and replace personally identifiable information (PII) in the text.
Replace each PII entity with a semantically similar but fake alternative that preserves context.
Always call replace_entities. If there is no PII, return an empty replacements list.

RULES:
• Names: replace with a different name of the same culture, gender, era.
• Companies / orgs: invent a fictitious org in the same industry and size.
• Projects / codenames: replace with a neutral two-word alias.
• Emails, phone numbers, IDs, keys: replace with format-valid fakes."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "replace_entities",
        "description": "Replace PII entities with anonymized versions",
        "parameters": {
            "type": "object",
            "properties": {
                "replacements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "replacement": {"type": "string"},
                        },
                        "required": ["original", "replacement"],
                    },
                }
            },
            "required": ["replacements"],
        },
    },
}]


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
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.text[:4000] + "\n/no_think"},
                ],
                "tools": TOOLS,
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        resp.raise_for_status()

    body = resp.json()
    message = body.get("message", {})
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        # Model found no PII — return text unchanged
        log.info("no PII detected")
        return AnonymizeResponse(anonymized=req.text, mapping={})

    arguments = tool_calls[0].get("function", {}).get("arguments", {})
    # Ollama native API returns arguments as a dict already (not a JSON string)
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    replacements = arguments.get("replacements", [])
    log.info("replacing %d entities", len(replacements))

    anonymized = req.text
    # mapping: replacement → original (for deanonymization)
    mapping: dict[str, str] = {}
    for pair in replacements:
        original = pair.get("original", "")
        replacement = pair.get("replacement", "")
        if original and replacement and original != replacement:
            anonymized = anonymized.replace(original, replacement)
            mapping[replacement] = original

    return AnonymizeResponse(anonymized=anonymized, mapping=mapping)


@app.post("/deanonymize", response_model=DeanonymizeResponse)
async def deanonymize(req: DeanonymizeRequest):
    text = req.text
    # Sort by length descending to avoid partial replacements (e.g. "DataSoft LLC" before "DataSoft")
    for replacement, original in sorted(req.mapping.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(replacement, original)
    return DeanonymizeResponse(text=text)


@app.get("/health")
async def health():
    return {"status": "ok"}
