# iterai_backend (VERIFYXO Backend)

## 1) Setup

```bash
cd "/Users/rashmi/iterai copy"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy env template and fill credentials:

```bash
cp .env.example .env
```

Required:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

Optional tuning:
- `LEXICAL_REJECT_THRESHOLD`
- `ENSEMBLE_REJECT_THRESHOLD`
- `VECTOR_MATCH_THRESHOLD`
- `VECTOR_MATCH_COUNT`
- `SEMANTIC_RPC_TIMEOUT_SECONDS`

## 2) Run API

```bash
cd "/Users/rashmi/iterai copy"
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Verify endpoint:

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"title":"Daily Morning Herald"}'
```

## 3) Seed Embeddings (one-time / refresh)

```bash
cd "/Users/rashmi/iterai copy"
source venv/bin/activate
python seed_vectors.py
```

## 4) Test & Lint

```bash
cd "/Users/rashmi/iterai copy"
source venv/bin/activate
pytest -q
ruff check .
```

## 5) Railway Deploy Notes

- Keep `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in Railway Variables.
- This repo's Dockerfile installs **CPU-only** PyTorch so image size stays under free-plan limits.
- Ensure Railway Start Command is empty (Dockerfile `CMD` is used), or set it to:

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```
