"""Run the agent on every case of a split and save traces + automatic metrics.

Automatic metrics (no grading): tool use, searches per question, Context Evidence Rate (gold evidence
inside the final context shown to the model), status counts, latency, tokens, cost.
Answer quality, citation support and abstention are graded by a human in results/agent/<split>/<config>/reviews.json.

Run: python -m src.run_agent --split dev --configs agent_dense_k3 ...
The test split only runs for configs listed in results/selection.json.
"""
import argparse
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .agent import Agent
from .common import AGENT_MODEL, RESULTS, is_hit, load_cases, load_chunks, load_json, load_questions, save_json
from .retrievers import Usage, build

AGENT_CONFIGS = {
    "agent_dense_k3": {"method": "dense", "top_k": 3, "max_searches": 3, "budget": 6000, "short_ids": True},
    "agent_bm25_k3": {"method": "bm25_bi", "top_k": 3, "max_searches": 3, "budget": 6000, "short_ids": True},
    "agent_hybrid_rerank_k3": {"method": "hybrid_rerank", "top_k": 3, "max_searches": 3, "budget": 6000, "short_ids": True},
    "agent_hybrid_rerank_k5": {"method": "hybrid_rerank", "top_k": 5, "max_searches": 3, "budget": 9000, "short_ids": True},
}


def run_config(name, split, workers):
    cfg = AGENT_CONFIGS[name]
    chunks = load_chunks()
    gold = {q["id"]: q["gold"] for q in load_questions(split)}
    cases = load_cases(split)

    def one(case):
        usage = Usage()
        agent = Agent(build(cfg["method"], chunks, usage), usage, top_k=cfg["top_k"],
                      max_searches=cfg["max_searches"], budget=cfg["budget"], short_ids=cfg["short_ids"])
        try:
            out = agent.run(case["question"], filters=case.get("filter"))
            err = None
        except Exception as e:  # provider/DB failure: recorded as error, excluded from quality metrics
            out, err = {"status": "error", "answer": "", "citations": [], "events": [], "n_searches": 0,
                        "context_ids": [], "context": "", "context_chars": 0, "elapsed_s": None}, f"{type(e).__name__}: {e}"
        ctx_hit = None
        if case["answerable"]:
            by_id = {c["id"]: c for c in chunks}
            ctx_hit = any(is_hit(by_id[i], gold[case["id"]]) for i in out["context_ids"])
        return {"query_id": case["id"], "split": split, "config": name, "model": AGENT_MODEL, **cfg,
                "question": case["question"], "answerable": case["answerable"], "error": err,
                "context_evidence": ctx_hit, **out, **usage.snapshot()}

    with ThreadPoolExecutor(workers) as ex:
        traces = list(ex.map(one, cases))
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    una = [t for t in ok if not t["answerable"]]
    summary = {
        "config": name, "split": split, "model": AGENT_MODEL, **cfg, "cases": len(traces),
        "errors": len(traces) - len(ok),
        "tool_use": f"{sum(t['n_searches'] > 0 for t in ok)}/{len(ok)}",
        "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "mean_searches": round(statistics.mean(t["n_searches"] for t in ok), 2) if ok else None,
        "context_evidence": f"{sum(t['context_evidence'] for t in ans)}/{len(ans)}",
        "answered_on_answerable": f"{sum(t['status'] == 'answered' for t in ans)}/{len(ans)}",
        "abstained_on_unanswerable": f"{sum(t['status'] == 'abstain' for t in una)}/{len(una)}",
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in ok)) if ok else None,
        "median_latency_s": statistics.median(t["elapsed_s"] for t in ok) if ok else None,
        "llm_input_tokens": sum(t["llm_input_tokens"] for t in traces),
        "llm_output_tokens": sum(t["llm_output_tokens"] for t in traces),
        "cost_usd": round(sum(t["cost_usd"] for t in traces), 5),
    }
    out_dir = RESULTS / "agent" / split / name
    save_json(out_dir / "traces.json", traces)
    save_json(out_dir / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="+", required=True, choices=list(AGENT_CONFIGS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.split == "test":
        sel_path = RESULTS / "selection.json"
        if not sel_path.exists():
            raise SystemExit("test is locked until results/selection.json records the configuration chosen on dev")
        allowed = set(load_json(sel_path)["run_on_test"])
        if not set(args.configs) <= allowed:
            raise SystemExit(f"only {sorted(allowed)} may run on test")
    for name in args.configs:
        s = run_config(name, args.split, args.workers)
        print({k: s[k] for k in ("config", "errors", "tool_use", "searches_distribution", "context_evidence",
                                 "answered_on_answerable", "abstained_on_unanswerable", "median_latency_s", "cost_usd")})


if __name__ == "__main__":
    main()
