# llm-anonymizer

A privacy-preserving text transformer for LLM pipelines. Replaces sensitive entities with placeholder tokens before sending to a cloud API, then restores them from the response — so the real data never leaves your cluster.

```
Input text → /anonymize → [tokens] → Cloud LLM → response with [tokens] → /deanonymize → restored text
```

Built to run on Kubernetes alongside a local llama.cpp inference server. No GPU required.

## How it works

**Anonymize** (`POST /anonymize`):
1. Sends text to a local model (llama.cpp) with a structured extraction prompt
2. Model identifies sensitive entities and replaces each with a typed token (`[PERSON_1]`, `[ORG_1]`, `[EMAIL_1]`, etc.)
3. Returns `{ anonymized_text, mapping }` — the mapping is what you'll use to restore

**Send to cloud**:
- Use the anonymized text in your cloud LLM request
- Include a system prompt instructing the model to preserve bracketed tokens exactly as written

**Deanonymize** (`POST /deanonymize`):
- Pure string substitution — no LLM call needed
- Pass the response text and the original mapping
- Tokens are replaced back with real values

## Entity types detected

| Token prefix | Detected entity |
|---|---|
| `[PERSON_N]` | People's names |
| `[ORG_N]` | Company / organization names |
| `[PROJECT_N]` | Project / product names |
| `[EMAIL_N]` | Email addresses |
| `[IP_N]` | IP addresses |
| `[HOST_N]` | Hostnames / internal URLs |
| `[CRED_N]` | Credentials, tokens, API keys |
| `[PHONE_N]` | Phone numbers |
| `[ADDR_N]` | Physical addresses |

## API

| Endpoint | Description |
|---|---|
| `POST /anonymize` | Replace entities with tokens. Body: `{ "text": "..." }`. Returns `{ "anonymized": "...", "mapping": {...} }` |
| `POST /deanonymize` | Restore tokens. Body: `{ "text": "...", "mapping": {...} }`. Returns `{ "text": "..." }` |
| `GET /health` | Liveness probe |

## Example

```bash
# Anonymize
curl -s -X POST http://localhost:8080/anonymize \
  -H 'Content-Type: application/json' \
  -d '{"text": "John Smith from ACME Corp needs help fixing the OAuth integration on auth.acme.internal"}' | jq

# Response:
{
  "anonymized": "[PERSON_1] from [ORG_1] needs help fixing the [PROJECT_1] integration on [HOST_1]",
  "mapping": {
    "[PERSON_1]": "John Smith",
    "[ORG_1]": "ACME Corp",
    "[PROJECT_1]": "OAuth",
    "[HOST_1]": "auth.acme.internal"
  }
}

# Send anonymized text to cloud LLM (with system prompt to preserve tokens)
# ... cloud response: "[PERSON_1] should check the [PROJECT_1] docs for [ORG_1]'s [HOST_1] config"

# Deanonymize
curl -s -X POST http://localhost:8080/deanonymize \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "[PERSON_1] should check the [PROJECT_1] docs for [ORG_1]'\''s [HOST_1] config",
    "mapping": {"[PERSON_1]": "John Smith", "[ORG_1]": "ACME Corp", "[PROJECT_1]": "OAuth", "[HOST_1]": "auth.acme.internal"}
  }' | jq

# Response:
{
  "text": "John Smith should check the OAuth docs for ACME Corp's auth.acme.internal config"
}
```

## Configuration

| Env var | Required | Description |
|---|---|---|
| `LLAMA_BASE_URL` | yes | Local llama.cpp server base URL, e.g. `http://llama-server:8080/v1` |

## n8n integration

An example n8n workflow is included in [`n8n/workflow-anonymizer-pipeline.json`](n8n/workflow-anonymizer-pipeline.json).

It implements a full pipeline triggered by webhook:
1. Receive `{ text, model }` via POST
2. Anonymize via this service
3. Send anonymized text to NVIDIA NIM (via ai-guard)
4. Deanonymize the response
5. Return clean result

Import the JSON directly into n8n via Settings → Import workflow. Wire any upstream source (Jira webhook, Slack slash command, cron job) to the webhook trigger.

## Kubernetes deployment

See [`k8s/`](k8s/) for ready-to-apply manifests. Deploy in the same namespace as your llama.cpp inference server for intra-namespace connectivity (no network policy changes needed).

## Caveats

- **Anonymization quality depends on the local model.** Smaller models (Phi-3.5-mini) may miss subtle entities or over-anonymize generic terms. Review results for your use case.
- **Token preservation by cloud model is not guaranteed.** The system prompt instructs the model to keep tokens intact, but a heavily rephrased response may drop or alter them. Test with your target model.
- **Not a compliance tool.** This is a best-effort privacy layer, not a certified data masking solution.

## Image

```
ghcr.io/janos-gyorgy/llm-anonymizer:latest
```

Built automatically on push to `main` via GitHub Actions.

## License

MIT — Janos Gyorgy
