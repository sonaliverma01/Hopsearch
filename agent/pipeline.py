"""The multi-hop search agent.

Bridge question:
    hop 1 : rank paragraphs for the full question (tf-idf cosine, zones, top-K)
    bridge: find entity names in the top hop-1 paragraphs that are titles of
            other paragraphs (dictionary lookup)
    hop 2 : query = bridge entity + RESIDUAL question terms (terms the hop-1
            paragraph did not already cover), e.g. "Christopher Nolan" + "univers studi"
    fuse  : chain score = alpha * s1 + (1 - alpha) * s2 * conf, ranked chains are
            flattened into one ranked list of paragraphs

Comparison question:
    one sub-query per entity, results fused with Reciprocal Rank Fusion (RRF)

Every step is recorded in `trail`, so the user can see exactly why each paragraph
was returned (transparency / verifiability).
"""
from collections import defaultdict

from agent.bridge import find_bridges
from agent.decompose import decompose
from ir.preprocess import terms


def rrf(lists, k=60, weights=None):
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d))."""
    scores = defaultdict(float)
    weights = weights or [1.0] * len(lists)
    for lst, w in zip(lists, weights):
        for rank, r in enumerate(lst, 1):
            scores[r["doc"]] += w / (k + rank)
    return sorted(scores, key=lambda d: -scores[d]), scores


class MultiHopAgent:
    def __init__(self, scorer, alpha=0.5, k1=10, top_p=3, max_bridges=3, fusion="chain"):
        self.s = scorer
        self.idx = scorer.idx
        self.alpha, self.k1, self.top_p = alpha, k1, top_p
        self.max_bridges, self.fusion = max_bridges, fusion

    def search(self, question, k=10):
        plan = decompose(question)
        if plan["type"] == "comparison":
            return self._comparison(question, plan, k)
        return self._bridge(question, plan, k)

    # ---------------------------------------------------------------- bridge
    def _bridge(self, question, plan, k):
        idx = self.idx
        hop1 = self.s.search(question, self.k1)
        trail = [{"step": "hop1", "query": question, "results": hop1[:5]}]
        if not hop1:
            return {"plan": plan, "ranking": [], "trail": trail}
        bridges = find_bridges(idx, hop1, question, self.top_p, self.max_bridges)
        trail.append({"step": "bridges", "bridges": bridges})
        q_terms = terms(question, idx.stem, idx.stop)
        max1 = hop1[0]["score"] or 1.0
        s1 = {r["doc"]: r["score"] / max1 for r in hop1}

        # run hop 2 for every bridge first, then normalise hop-2 scores by the
        # GLOBAL max, so a bridge whose paragraph also matches the residual words
        # ("born", "university") beats a bridge that only matches its own name.
        hop2_runs = []
        for b in bridges:
            p1 = b["from_doc"]
            p1_terms = set(terms(idx.docs[p1]["title"] + " " + idx.docs[p1]["text"],
                                 idx.stem, idx.stop))
            residual = [t for t in q_terms if t not in p1_terms]
            ent_terms = terms(idx.docs[b["doc"]]["title"], idx.stem, idx.stop)
            q2 = ent_terms + residual
            hop2 = self.s.search_terms(q2, 5, exclude={p1})
            hop2_runs.append((b, p1, hop2))
            trail.append({"step": "hop2", "bridge": b["entity"],
                          "bridge_title": idx.docs[b["doc"]]["title"],
                          "query_terms": q2, "residual": residual, "results": hop2})
        max2 = max((r["score"] for _, _, h in hop2_runs for r in h), default=1.0) or 1.0

        chains, hop2_lists = [], []
        for b, p1, hop2 in hop2_runs:
            hop2_lists.append(hop2)
            for r in hop2:
                # the bridge paragraph itself gets its full lookup confidence
                conf = b["conf"] if r["doc"] == b["doc"] else 0.5 * b["conf"]
                chain = self.alpha * s1.get(p1, 0) + (1 - self.alpha) * (r["score"] / max2) * conf
                chains.append((chain, p1, r["doc"], b["entity"]))
            if b["doc"] not in {r["doc"] for r in hop2}:
                chains.append((self.alpha * s1.get(p1, 0) + (1 - self.alpha) * 0.3 * b["conf"],
                               p1, b["doc"], b["entity"]))

        if self.fusion == "rrf":
            order, _ = rrf([hop1] + hop2_lists, weights=[1.0] + [1.0] * len(hop2_lists))
        else:
            chains.sort(key=lambda c: -c[0])
            order, seen = [], set()
            for _, p1, p2, _e in chains:
                for d in (p1, p2):
                    if d not in seen:
                        seen.add(d); order.append(d)
            for r in hop1:
                if r["doc"] not in seen:
                    seen.add(r["doc"]); order.append(r["doc"])
        trail.append({"step": "fusion", "method": self.fusion,
                      "chains": [{"score": c[0], "p1": c[1], "p2": c[2], "bridge": c[3]}
                                 for c in sorted(chains, key=lambda c: -c[0])[:5]]})
        return {"plan": plan, "ranking": order[:k], "trail": trail}

    # ------------------------------------------------------------ comparison
    def _comparison(self, question, plan, k):
        lists, trail = [], []
        for sq, ent in zip(plan["subqueries"], plan["entities"]):
            # entity words are repeated so the entity dominates the sub-query
            res = self.s.search(f"{ent} {ent} {sq}", self.k1)
            lists.append(res)
            trail.append({"step": "subquery", "entity": ent, "query": sq, "results": res[:5]})
        full = self.s.search(question, self.k1)
        lists.append(full)
        trail.append({"step": "full_question", "query": question, "results": full[:5]})
        order, _ = rrf(lists, weights=[1.0] * (len(lists) - 1) + [0.5])
        trail.append({"step": "fusion", "method": "rrf"})
        return {"plan": plan, "ranking": order[:k], "trail": trail}


def baseline(scorer, question, k=10):
    """Single-hop baseline: the whole question as one tf-idf query."""
    return [r["doc"] for r in scorer.search(question, k)]
