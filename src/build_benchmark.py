"""Build and verify the Day 7 benchmark.

dev  = all 26 Day 6 questions (Day 6 dev + Day 6 test) + 3 unanswerable Day 7 questions
test = 8 new answerable + 3 unanswerable questions written for Day 7 (data/benchmark/day7_additions.json)

Day 6 test was moved into dev because Day 7 builds a new system; a fresh test set keeps the
held-out evaluation independent. Verification: every gold evidence must lie inside one exported
chunk with the right source and page, otherwise the script exits with an error.

Run: python -m src.build_benchmark [--day6 <path to Day 6 benchmark/questions.jsonl>]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .common import CASES_PATH, DATA, QUESTIONS_PATH, ROOT, is_hit, load_chunks, save_json

DAY6_DEFAULT = ROOT.parent / "K5-L3B-DAY06-2A202603896-DataIngestion" / "benchmark" / "questions.jsonl"


def convert_day6(q: dict) -> dict:
    gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
    gold += [{"source_id": a["gold_source_id"], "page": a["gold_page"], "evidence": a["evidence_quote"]}
             for a in q.get("alt_gold") or []]
    out = {"id": q["id"], "split": "dev", "day6_split": q["split"], "type": q.get("type"),
           "question": q["question"], "expected_answer": q["expected_answer"], "gold": gold}
    if q.get("filter"):
        out["filter"] = q["filter"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day6", type=Path, default=DAY6_DEFAULT)
    args = ap.parse_args()
    day6 = [json.loads(l) for l in args.day6.read_text(encoding="utf-8").splitlines() if l.strip()]
    add = json.loads((DATA / "benchmark" / "day7_additions.json").read_text(encoding="utf-8"))

    questions = [convert_day6(q) for q in day6] + add["answerable"]
    unanswerable = [{**u, "answerable": False, "expected_behavior": "search_and_abstain"} for u in add["unanswerable"]]

    chunks = load_chunks()
    errors = []
    ids = Counter(q["id"] for q in questions + unanswerable)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for q in questions:
        for g in q["gold"]:
            if not any(is_hit(c, [g]) for c in chunks):
                errors.append(f"{q['id']}: evidence on p{g['page']} not inside one chunk: {g['evidence'][:60]}")
    if errors:
        raise SystemExit("\n".join(errors))

    cases = [{"id": q["id"], "split": q["split"], "question": q["question"], "answerable": True,
              "expected_behavior": "search_and_answer", **({"filter": q["filter"]} if q.get("filter") else {})}
             for q in questions] + [{k: u[k] for k in ("id", "split", "question", "answerable",
                                                        "expected_behavior", "absence_reason")}
                                    for u in unanswerable]
    save_json(QUESTIONS_PATH, questions)
    save_json(CASES_PATH, cases)
    print(Counter((c["split"], c["answerable"]) for c in cases))


if __name__ == "__main__":
    main()
