import os
from contextlib import asynccontextmanager
from functools import lru_cache
from threading import RLock
from typing import List, Optional, Tuple

import jellyfish
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langdetect import LangDetectException, detect
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from embeddings import embed
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from title_engine import TitleIndex, enforce_guidelines, sanitize_input


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials in .env")

LEXICAL_REJECT_THRESHOLD = _env_float("LEXICAL_REJECT_THRESHOLD", 82.0)
ENSEMBLE_REJECT_THRESHOLD = _env_float("ENSEMBLE_REJECT_THRESHOLD", 70.0)
VECTOR_MATCH_THRESHOLD = _env_float("VECTOR_MATCH_THRESHOLD", 0.35)
VECTOR_MATCH_COUNT = _env_int("VECTOR_MATCH_COUNT", 5)
SEMANTIC_RPC_TIMEOUT_SECONDS = _env_float("SEMANTIC_RPC_TIMEOUT_SECONDS", 3.0)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
semantic_supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
    options=SyncClientOptions(postgrest_client_timeout=SEMANTIC_RPC_TIMEOUT_SECONDS),
)
title_index = TitleIndex()
index_lock = RLock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Bootstrapping in-memory index from Supabase...")
    batch_size = 1000
    start = 0
    loaded = 0
    while True:
        response = (
            supabase.table("existing_titles")
            .select("Title")
            .range(start, start + batch_size - 1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            break

        batch_titles = [
            raw_title
            for row in rows
            if (raw_title := (row.get("Title") or row.get("title")))
        ]

        with index_lock:
            title_index.extend(batch_titles)
        loaded += len(batch_titles)

        start += batch_size
        if loaded % 10000 == 0:
            print(f"Indexed {loaded} titles...")

    print(f"Index ready: {loaded} titles.")
    yield
    with index_lock:
        title_index.clear()


app = FastAPI(title="VERIFYXO Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TitleSubmission(BaseModel):
    title: str
    language: Optional[str] = None


class VerificationResponse(BaseModel):
    status: str
    verification_probability: float
    similarity_percentage: float
    is_rejected: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    feedback: str


class OfficialApplication(BaseModel):
    title: str
    language: str
    owner_email: str


class WebhookPayload(BaseModel):
    title: str
    government_registration_id: str
    status: str


INDIC_LANG_MAP = {
    "hi": "Hindi",
    "en": "English",
    "bn": "Bengali",
    "mr": "Marathi",
    "te": "Telugu",
    "ta": "Tamil",
    "gu": "Gujarati",
    "ur": "Urdu",
    "kn": "Kannada",
    "or": "Odia",
    "ml": "Malayalam",
    "pa": "Punjabi",
}


def auto_detect_language(text: str) -> str:
    stripped = (text or "").strip()
    if len(stripped) < 4:
        return "English"
    try:
        code = detect(stripped)
        return INDIC_LANG_MAP.get(code, "English")
    except LangDetectException:
        return "English"


@lru_cache(maxsize=20000)
def _cached_metaphone(text: str) -> str:
    return jellyfish.metaphone(text)


def check_combinations_and_phonetics(title: str) -> Tuple[List[str], float, str]:
    normalized = sanitize_input(title)
    with index_lock:
        reasons, score = title_index.detect_lexical_conflicts(normalized, precleaned=True)

    if reasons and normalized and normalized != title.lower().strip():
        reasons = [f"Input was normalized to '{normalized}' before matching."] + reasons
    return reasons, score, normalized


def _build_ensemble_reason(
    matched_title: str,
    total_similarity: float,
    semantic_score: float,
    phonetic_score: float,
    fuzzy_score: float,
) -> str:
    weighted = {
        "Similarity in meaning (semantic conflict)": 0.40 * semantic_score,
        "Similarity in sound (phonetic conflict)": 0.35 * phonetic_score,
        "Similarity in spelling (lexical conflict)": 0.25 * fuzzy_score,
    }
    primary = max(weighted, key=weighted.get)
    return (
        f"{primary} with existing title '{matched_title}' "
        f"({total_similarity:.1f}% total similarity)."
    )


@lru_cache(maxsize=5000)
def cached_verification_logic(title: str, language: str) -> VerificationResponse:
    lexical_rejections, lexical_score, clean_title = check_combinations_and_phonetics(title)
    if lexical_score >= LEXICAL_REJECT_THRESHOLD:
        probability = max(0.0, 100.0 - lexical_score)
        return VerificationResponse(
            status="rejected",
            verification_probability=round(probability, 2),
            similarity_percentage=round(lexical_score, 2),
            is_rejected=True,
            rejection_reasons=lexical_rejections,
            feedback="Title is too close to an existing title by lexical/phonetic checks.",
        )

    with index_lock:
        rule_rejections = enforce_guidelines(clean_title, title_index, precleaned=True)
    if rule_rejections:
        return VerificationResponse(
            status="rejected",
            verification_probability=0.0,
            similarity_percentage=100.0,
            is_rejected=True,
            rejection_reasons=rule_rejections,
            feedback="Title violates PRGI naming guidelines.",
        )

    if lexical_score >= ENSEMBLE_REJECT_THRESHOLD:
        probability = max(0.0, 100.0 - lexical_score)
        reasons = lexical_rejections or [
            (
                "Lexical similarity is already above rejection threshold "
                f"({lexical_score:.1f}% >= {ENSEMBLE_REJECT_THRESHOLD:.1f}%)."
            )
        ]
        return VerificationResponse(
            status="rejected",
            verification_probability=round(probability, 2),
            similarity_percentage=round(lexical_score, 2),
            is_rejected=True,
            rejection_reasons=reasons,
            feedback="Rejected by lexical scoring without semantic stage.",
        )

    highest_ensemble_score = 0.0
    ensemble_reasons: List[str] = []
    try:
        raw_vector = embed(title)
        # HuggingFace might return [vector] instead of vector for single string
        if isinstance(raw_vector, list) and len(raw_vector) > 0 and isinstance(raw_vector[0], list):
            query_vector = raw_vector[0]
        else:
            query_vector = raw_vector
            
        rpc_response = semantic_supabase.rpc(
            "match_titles",
            {
                "query_embedding": query_vector,
                "match_threshold": VECTOR_MATCH_THRESHOLD,
                "match_count": VECTOR_MATCH_COUNT,
            },
        ).execute()
        matches = rpc_response.data or []
        seen_matches = set()
        query_metaphone = _cached_metaphone(clean_title)

        for match in matches:
            matched_title = match.get("Title") or match.get("title")
            if not matched_title or matched_title in seen_matches:
                continue
            seen_matches.add(matched_title)

            clean_match = sanitize_input(matched_title)
            if not clean_match:
                continue
            semantic_score = max(0.0, min(100.0, float(match.get("similarity", 0)) * 100.0))
            phonetic_score = 0.0
            if query_metaphone and query_metaphone == _cached_metaphone(clean_match):
                phonetic_score = 100.0
            fuzzy_score = float(fuzz.ratio(clean_title, clean_match))

            total_similarity = (
                (0.40 * semantic_score)
                + (0.35 * phonetic_score)
                + (0.25 * fuzzy_score)
            )
            highest_ensemble_score = max(highest_ensemble_score, total_similarity)

            if total_similarity >= ENSEMBLE_REJECT_THRESHOLD:
                ensemble_reasons.append(
                    _build_ensemble_reason(
                        matched_title=matched_title,
                        total_similarity=total_similarity,
                        semantic_score=semantic_score,
                        phonetic_score=phonetic_score,
                        fuzzy_score=fuzzy_score,
                    )
                )
                break

    except Exception as exc:
        print(f"Semantic RPC failed, continuing without vector stage: {exc}")

    final_similarity = max(lexical_score, highest_ensemble_score)
    probability = max(0.0, 100.0 - final_similarity)

    if final_similarity >= ENSEMBLE_REJECT_THRESHOLD:
        reasons = list(dict.fromkeys(lexical_rejections + ensemble_reasons))
        if not reasons:
            reasons = ["High conceptual similarity detected with existing registered titles."]
        return VerificationResponse(
            status="rejected",
            verification_probability=round(probability, 2),
            similarity_percentage=round(final_similarity, 2),
            is_rejected=True,
            rejection_reasons=reasons,
            feedback="Rejected by weighted lexical, phonetic, and semantic scoring.",
        )

    return VerificationResponse(
        status="success",
        verification_probability=round(probability, 2),
        similarity_percentage=round(final_similarity, 2),
        is_rejected=False,
        rejection_reasons=[],
        feedback="Title passed automated validation checks.",
    )


@app.get("/healthz")
async def healthz():
    with index_lock:
        total_titles = len(title_index.existing_titles)
    return {"status": "ok", "indexed_titles": total_titles}


@app.post("/verify", response_model=VerificationResponse)
async def verify_new_title(submission: TitleSubmission):
    if not submission.language:
        submission.language = auto_detect_language(submission.title)
    return cached_verification_logic(submission.title, submission.language)


@app.post("/submit_application")
async def submit_application(app_data: OfficialApplication):
    clean_title = sanitize_input(app_data.title)
    if not clean_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")

    with index_lock:
        if clean_title in title_index.existing_titles:
            raise HTTPException(status_code=409, detail="Title already exists.")

    try:
        supabase.table("existing_titles").insert(
            {
                "Title": app_data.title,
                "Language": app_data.language,
                "Publication State": "pending",
            }
        ).execute()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to persist application: {exc}"
        ) from exc

    with index_lock:
        title_index.add_title(app_data.title)
    cached_verification_logic.cache_clear()
    return {"status": "success", "message": "Official application submitted to PRGI."}


@app.post("/webhook/prgi_sync")
async def prgi_sync_webhook(payload: WebhookPayload):
    try:
        supabase.table("existing_titles").update(
            {
                "Publication State": payload.status,
                "PRGI_Reg_ID": payload.government_registration_id,
            }
        ).eq("Title", payload.title).execute()
        return {"status": "success", "message": "Database synced with official PRGI records."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sync failed: {exc}") from exc
