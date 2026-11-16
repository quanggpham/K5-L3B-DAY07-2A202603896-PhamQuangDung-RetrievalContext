"""Offline checks for the Day 7 pipeline (src/). Fake clients test control flow only, not quality."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import Agent, normalize_markers  # noqa: E402
from src.common import is_hit, tokenize  # noqa: E402
from src.retrievers import BM25Retriever, HybridRRF, Usage  # noqa: E402


def chunk(cid, text, page=1, section="s1"):
    return {"id": cid, "text": text, "source_id": "doc", "source": "doc.pdf", "page": page,
            "metadata": {"section": section}}


CHUNKS = [chunk("a", "Tăng trưởng GDP đạt 3,7% trong nửa đầu năm 2023."),
          chunk("b", "Nợ xấu tăng từ 1,9% lên 2,9%.", section="s2"),
          chunk("c", "Đầu tư công tăng 43,3% so cùng kỳ.")]


class Fixed:
    def __init__(self, name, ids):
        self.name, self.ids = name, ids

    def search(self, query, *, top_k, filters=None):
        by = {c["id"]: c for c in CHUNKS}
        return [{**by[i], "rank": r + 1, "score": 1.0} for r, i in enumerate(self.ids[:top_k])]


class RetrievalTests(unittest.TestCase):
    def test_tokenize_keeps_decimal_numbers(self):
        self.assertIn("3,7", tokenize("GDP đạt 3,7%"))

    def test_bm25_ranks_exact_terms_and_applies_filter(self):
        bm = BM25Retriever(CHUNKS)
        self.assertEqual(bm.search("nợ xấu", top_k=3)[0]["id"], "b")
        self.assertEqual(bm.search("nợ xấu", top_k=3, filters={"section": "s1"}), [])
        with self.assertRaises(ValueError):
            bm.search("x", top_k=1, filters={"year": 2023})

    def test_rrf_rewards_agreement(self):
        out = HybridRRF(Fixed("d", ["a", "c", "b"]), Fixed("l", ["c", "a", "b"]), depth=3).search("q", top_k=3)
        self.assertEqual({out[0]["id"], out[1]["id"]}, {"a", "c"})
        self.assertEqual(out[2]["id"], "b")
        self.assertEqual([r["rank"] for r in out], [1, 2, 3])

    def test_is_hit_ignores_layout_whitespace_but_checks_page(self):
        gold = [{"source_id": "doc", "page": 1, "evidence": "đạt  3,7%\ntrong"}]
        self.assertTrue(is_hit(CHUNKS[0], gold))
        self.assertFalse(is_hit({**CHUNKS[0], "page": 2}, gold))


def tool_msg(name, args, i):
    return NS(content=None, tool_calls=[NS(id=f"t{i}", function=NS(name=name, arguments=json.dumps(args, ensure_ascii=False)),
                                           model_dump=lambda i=i, n=name, a=args: {"id": f"t{i}", "type": "function",
                                                                                   "function": {"name": n, "arguments": json.dumps(a)}})])


class FakeClient:
    """Replays scripted tool calls and records what the agent sent."""

    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.sent.append(kw)
        name, args = self.script.pop(0)
        return NS(usage=NS(prompt_tokens=10, completion_tokens=2),
                  choices=[NS(message=tool_msg(name, args, len(self.sent)))])


class AgentTests(unittest.TestCase):
    def run_agent(self, script, ids=("a", "b"), **kw):
        client = FakeClient(script)
        agent = Agent(Fixed("fixed", list(ids)), Usage(), client=client, **kw)
        return agent.run("GDP?"), client

    def test_short_labels_are_mapped_back_to_chunk_ids(self):
        out, _ = self.run_agent([("search_report", {"query": "GDP"}),
                                 ("submit_answer", {"answer": "3,7% [S1, S2]", "citations": ["S1", "S2"], "abstain": False})])
        self.assertEqual(out["status"], "answered")
        self.assertEqual(out["answer"], "3,7% [a][b]")
        self.assertEqual(out["citations"], ["a", "b"])

    def test_citation_outside_context_abstains_after_one_repair(self):
        bad = ("submit_answer", {"answer": "3,7% [S9]", "citations": ["S9"], "abstain": False})
        out, _ = self.run_agent([("search_report", {"query": "GDP"}), bad, bad])
        self.assertEqual(out["status"], "abstain")
        self.assertEqual([e["event"] for e in out["events"]].count("invalid_citations"), 2)

    def test_search_limit_forces_submit_answer(self):
        s = ("search_report", {"query": "GDP"})
        out, client = self.run_agent([s, s, ("submit_answer", {"answer": "x", "citations": [], "abstain": True})], max_searches=2)
        self.assertEqual(out["status"], "abstain")
        self.assertEqual(out["n_searches"], 2)
        self.assertEqual(client.sent[-1]["tool_choice"], {"type": "function", "function": {"name": "submit_answer"}})

    def test_budget_and_dedup(self):
        s = ("search_report", {"query": "q"})
        out, _ = self.run_agent([s, s, ("submit_answer", {"answer": "x", "citations": [], "abstain": True})],
                                budget=len(CHUNKS[0]["text"]) + 40)
        searches = [e for e in out["events"] if e["event"] == "search"]
        self.assertEqual(out["context_ids"], ["a"])
        self.assertEqual(searches[0]["dropped_budget"], ["b"])
        self.assertEqual(searches[1]["repeated"], ["a"])

    def test_normalize_markers_only_rewrites_known_labels(self):
        known = {"S1": "a", "S2": "b"}
        self.assertEqual(normalize_markers("x [S1 | trang 2 | s1] y [S1; S2] z [ghi chú]", known),
                         "x [S1] y [S1][S2] z [ghi chú]")


if __name__ == "__main__":
    unittest.main()
