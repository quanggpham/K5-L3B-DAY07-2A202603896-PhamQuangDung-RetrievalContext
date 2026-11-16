"""Export the Day 6 chunk index (strategy B) from pgvector into the Day 7 chunk format.

Run: python -m src.export_day6
Output: data/day6/chunks.json (gitignored: derived from a PDF that may not be redistributed).
Only text and metadata are exported; vectors stay in pgvector and dense search queries the DB.
"""
import json

import psycopg

from .common import CHUNKS_PATH, DATABASE_URL, DAY6_STRATEGY, EMBEDDING_MODEL, save_json


def main() -> None:
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("""SELECT chunk_id, document_id, source, page, section, text, metadata, embedding_model
                       FROM chunks WHERE strategy = %s ORDER BY page, chunk_id""", (DAY6_STRATEGY,))
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"no chunks for strategy {DAY6_STRATEGY}; run Day 6 ingest first")
    models = {r[7] for r in rows}
    if models != {EMBEDDING_MODEL}:
        raise RuntimeError(f"index built with {models}, but EMBEDDING_MODEL={EMBEDDING_MODEL}")
    chunks = []
    for cid, doc, source, page, section, text, meta, _ in rows:
        meta = meta if isinstance(meta, dict) else json.loads(meta)
        chunks.append({"id": cid, "text": text, "source_id": doc, "source": source, "page": page,
                       "metadata": {**meta, "section": section, "day6_strategy": DAY6_STRATEGY}})
    save_json(CHUNKS_PATH, chunks)
    print(f"exported {len(chunks)} chunks ({DAY6_STRATEGY}, {EMBEDDING_MODEL}) -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
