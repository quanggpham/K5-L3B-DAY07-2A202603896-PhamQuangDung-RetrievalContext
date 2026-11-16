"""Benchmark retrieval methods with the ORIGINAL question as query (no rewriting).

Metrics over answerable questions only (unanswerable ones are not in the denominator):
  Hit@1, Evidence Hit@3, Hit@5, MRR@10  - ranking quality from a top-10 list
  Context Evidence Rate                 - evidence chunk inside the packed context
                                          (top `context_k` chunks, greedily packed into `budget` chars)

Run: python -m src.eval_retrieval --split dev [--configs dense bm25_uni ...]
The test split only runs after results/selection.json fixes the configuration.
"""
import argparse
import json
import statistics
import time

from .common import RESULTS, is_hit, load_chunks, load_questions, save_json
from .retrievers import Usage, build

# name -> (method, context_k, budget_chars). Budget counts chunk text + header line.
CONFIGS = {
    "dense_k3": ("dense", 3, 4000),
    "bm25_uni_k3": ("bm25_uni", 3, 4000),
    "bm25_bi_k3": ("bm25_bi", 3, 4000),
    "hybrid_k3": ("hybrid", 3, 4000),
    "hybrid_rerank_k3": ("hybrid_rerank", 3, 4000),
    "dense_k5": ("dense", 5, 6000),
    "hybrid_k5": ("hybrid", 5, 6000),
    "hybrid_rerank_k5": ("hybrid_rerank", 5, 6000),
}


def pack_context(results, budget):
    blocks, ids, used = [], [], 0
    for c in results:
        block = f"[{c['id']} | trang {c['page']} | {c['metadata'].get('section')}]\n{c['text']}"
        if used + len(block) <= budget:
            blocks.append(block)
            ids.append(c["id"])
            used += len(block) + 2
    return "\n\n".join(blocks), ids


def run(name, questions, chunks):
    method, context_k, budget = CONFIGS[name]
    usage = Usage()
    r = build(method, chunks, usage)
    traces, errors = [], 0
    for q in questions:
        t0 = time.perf_counter()
        try:
            res = r.search(q["question"], top_k=max(10, context_k), filters=q.get("filter"))
            err = None
        except Exception as e:  # recorded, not replaced by fake results
            res, err = [], f"{type(e).__name__}: {e}"
            errors += 1
        ms = (time.perf_counter() - t0) * 1000
        hits = [is_hit(c, q["gold"]) for c in res]
        ctx, ctx_ids = pack_context(res[:context_k], budget)
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        traces.append({
            "query_id": q["id"], "question": q["question"], "query": q["question"], "filter": q.get("filter"),
            "config": name, "method": method, "error": err, "elapsed_ms": round(ms, 1),
            "results": [{"rank": c["rank"], "id": c["id"], "page": c["page"], "section": c["metadata"].get("section"),
                         "score": c["score"], "hit": h, **({"base_rank": c["base_rank"]} if "base_rank" in c else {})}
                        for c, h in zip(res, hits)],
            "first_hit_rank": first, "hit1": first == 1, "hit3": bool(first and first <= 3),
            "hit5": bool(first and first <= 5), "context_ids": ctx_ids, "context_chars": len(ctx),
            "context_hit": any(h and c["id"] in ctx_ids for c, h in zip(res, hits))})
    n = len(traces)
    frac = lambda k: f"{sum(t[k] for t in traces)}/{n}"
    summary = {"config": name, "method": method, "context_k": context_k, "budget_chars": budget, "n": n,
               "hit1": frac("hit1"), "hit3": frac("hit3"), "hit5": frac("hit5"),
               "mrr10": round(sum(1 / t["first_hit_rank"] for t in traces if t["first_hit_rank"]) / n, 3),
               "context_evidence": frac("context_hit"),
               "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
               "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
               "errors": errors, **usage.snapshot()}
    return summary, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    if args.split == "test" and not (RESULTS / "selection.json").exists():
        raise SystemExit("test is locked until results/selection.json records the configuration chosen on dev")
    chunks = load_chunks()
    questions = load_questions(args.split)
    table = []
    for name in args.configs:
        summary, traces = run(name, questions, chunks)
        save_json(RESULTS / "retrieval" / args.split / f"{name}.json", {"summary": summary, "traces": traces})
        table.append(summary)
        print(f"{name:18} hit1 {summary['hit1']:>6} hit3 {summary['hit3']:>6} hit5 {summary['hit5']:>6} "
              f"mrr {summary['mrr10']:.3f} ctx {summary['context_evidence']:>6} "
              f"{summary['median_ms']:>7}ms ${summary['cost_usd']:.5f} err {summary['errors']}")
    out = RESULTS / "retrieval" / f"{args.split}_summary.json"
    old = {s["config"]: s for s in (json.loads(out.read_text("utf-8")) if out.exists() else [])}
    old.update({s["config"]: s for s in table})
    save_json(out, list(old.values()))


if __name__ == "__main__":
    main()
