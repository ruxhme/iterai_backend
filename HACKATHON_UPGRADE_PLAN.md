# Hackathon Upgrade Plan (PRGI Title Validation)

## What is implemented now
- Faster lexical retrieval with in-memory candidate pruning (token + trigram indexes).
- Better collision handling (phonetic, acronym, and word-order maps now store multiple matches).
- New rule coverage:
  - Existing-title combination detection (e.g., `Hindu` + `Indian Express`).
  - Periodicity-extension rejection (e.g., `Daily <existing-title>`).
  - Prefix/suffix conflict checks tied to actual existing-title overlap.
- Cache invalidation after `/submit_application` so new pending titles are immediately enforced.
- Health endpoint: `/healthz`.
- CI/CD baseline:
  - Lint + tests in GitHub Actions.
  - Dockerfile for reproducible deployment.
  - Unit tests for core title-matching logic.

## High-impact next steps (for winning demo)
1. Add retrieval fusion (RRF):
   - Combine pgvector semantic neighbors + lexical candidate list with reciprocal rank fusion.
   - Improves recall for transliteration and paraphrase conflicts.
2. Add a calibrated risk score:
   - Train a small calibrator (isotonic/logistic) on labeled accepted/rejected history.
   - Convert raw similarity to reliable verification probability.
3. Add event-driven architecture with Kafka:
   - Publish events from `/verify` and `/submit_application` (`title.verified`, `title.submitted`).
   - Consumers: analytics, abuse detection, audit trail.
4. Add observability:
   - Per-request timings (normalization, lexical stage, vector stage).
   - Metrics dashboard with p95 latency and rejection reason distribution.
5. Add evaluation harness:
   - Gold test set with hard negatives/positives.
   - Track precision/recall at thresholds and keep under 2 seconds p95.

## Suggested architecture for demo slide
- `API (FastAPI)` -> `Lexical Index (RAM)` + `pgvector (Supabase)` -> `Decision Engine`.
- Emit decision events to `Kafka`.
- `Consumer 1`: real-time dashboard.
- `Consumer 2`: compliance audit log.
- `Consumer 3`: model retraining dataset builder.
