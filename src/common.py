"""Shared paths, environment, IO helpers and price table for the Day 7 pipeline."""
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHUNKS_PATH = DATA / "day6" / "chunks.json"          # exported from pgvector, gitignored
QUESTIONS_PATH = DATA / "benchmark" / "questions.json"
CASES_PATH = DATA / "benchmark" / "cases.json"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"

load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
EMBEDDING_DIM = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}[EMBEDDING_MODEL]
AGENT_MODEL = os.getenv("AGENT_MODEL") or "gpt-4.1-mini"
DAY6_STRATEGY = "B"   # chunking chosen on Day 6 dev (paragraph, 300-1000 chars)
SOURCE_ID = "wb-vn-taking-stock-2023-08-vi"

# USD per 1M tokens (OpenAI list price, checked 2026-11-16; update if it changes).
PRICES = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def flat(text: str) -> str:
    """Whitespace/Unicode normalisation used for evidence matching (layout line breaks carry no meaning)."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def tokenize(text: str) -> list[str]:
    """Vietnamese syllable tokens; numbers keep their decimal comma (e.g. '1,7')."""
    return re.findall(r"\d+(?:[.,]\d+)*|\w+", unicodedata.normalize("NFC", text).casefold())


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} missing; run `python -m src.export_day6` first")
    return load_json(CHUNKS_PATH)


def load_questions(split: str | None = None) -> list[dict[str, Any]]:
    qs = load_json(QUESTIONS_PATH)
    return [q for q in qs if split is None or q["split"] == split]


def load_cases(split: str | None = None) -> list[dict[str, Any]]:
    cs = load_json(CASES_PATH)
    return [c for c in cs if split is None or c["split"] == split]


def is_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """A chunk hits when source and page match and one evidence quote lies inside its text."""
    text = flat(chunk["text"])
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and flat(g["evidence"]) in text for g in gold)
