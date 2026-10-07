"""Ranked retrieval (Lectures 6 and 7).

score(q, d) = w_title * cos_title(q, d) + w_body * cos_body(q, d) [+ lam * g(d)]

cos_zone uses SMART lnc.ltc:
    query  : l (1 + log10 tf)  t (idf = log10 N/df)  c (cosine normalise)
    doc    : l (1 + log10 tf)  n (no idf)            c (cosine normalise, precomputed)

Computed term-at-a-time with score accumulators, then heap-based top-K
(heapq.nlargest) instead of sorting every score.

Efficiency options (Lecture 7):
    mode="full"      score every doc in every query term's postings
    mode="idf_elim"  index elimination: only query terms with idf >= idf_min,
                     and only docs containing >= 2 query terms
    mode="champion"  only score docs in each term's champion list

Other scoring functions for the comparison experiment:
    method="tf"       raw term-frequency overlap, no idf, no normalisation
    method="jaccard"  Jaccard coefficient between query and doc term sets (Lecture 6)
"""
import heapq
import math
from collections import Counter, defaultdict

from ir.preprocess import analyze


class Scorer:
    def __init__(self, idx, w_title=0.4, w_body=0.6, lam=0.0, method="lnc.ltc",
                 mode="full", idf_min=1.0):
        self.idx = idx
        self.w = {"title": w_title, "body": w_body}
        self.lam, self.method, self.mode, self.idf_min = lam, method, mode, idf_min

    def query_terms(self, text):
        return [t for t, _ in analyze(text, self.idx.stem, self.idx.stop)]

    def query_vector(self, terms):
        """ltc query weights."""
        tf = Counter(t for t in terms if t in self.idx.df)
        w = {t: (1 + math.log10(c)) * self.idx.idf(t) for t, c in tf.items()}
        n = math.sqrt(sum(v * v for v in w.values())) or 1.0
        return {t: v / n for t, v in w.items()}

    def search(self, text, k=10):
        return self.search_terms(self.query_terms(text), k)

    def search_terms(self, terms, k=10, exclude=()):
        idx = self.idx
        if self.method == "jaccard":
            return self._jaccard(terms, k, exclude)
        qv = self.query_vector(terms)
        if self.mode == "idf_elim":
            kept = {t: v for t, v in qv.items() if idf_ok(idx, t, self.idf_min)}
            qv = kept or qv   # never eliminate everything
        acc_t, acc_b = defaultdict(float), defaultdict(float)
        n_match = defaultdict(int)
        matched = defaultdict(list)
        for t, wq in qv.items():
            plist = idx.postings[t]
            docs = idx.champions[t] if self.mode == "champion" else plist
            for d in docs:
                tp, bp = plist[d]
                n_match[d] += 1
                matched[d].append(t)
                if self.method == "tf":
                    acc_t[d] += len(tp)
                    acc_b[d] += len(bp)
                    continue
                if tp:
                    acc_t[d] += wq * (1 + math.log10(len(tp))) / idx.norm["title"][d]
                if bp:
                    acc_b[d] += wq * (1 + math.log10(len(bp))) / idx.norm["body"][d]
        min_terms = 2 if (self.mode == "idf_elim" and len(qv) >= 2) else 1
        scores = {}
        for d in n_match:
            if d in exclude or n_match[d] < min_terms:
                continue
            scores[d] = (self.w["title"] * acc_t[d] + self.w["body"] * acc_b[d]
                         + self.lam * idx.g[d])
        top = heapq.nlargest(k, scores.items(), key=lambda x: x[1])  # heap top-K
        return [{"doc": d, "score": s,
                 "title_cos": acc_t[d], "body_cos": acc_b[d], "g": idx.g[d],
                 "matched": matched[d]} for d, s in top]

    def _doc_set(self, d):
        cache = self.idx.__dict__.setdefault("_docsets", {})
        if d not in cache:
            doc = self.idx.docs[d]
            cache[d] = {t for t, _ in analyze(doc["title"] + " " + doc["text"],
                                              self.idx.stem, self.idx.stop)}
        return cache[d]

    def _jaccard(self, terms, k, exclude):
        q = set(terms)
        cand = set()
        for t in q:
            cand |= set(self.idx.postings.get(t, {}))
        scores = {}
        for d in cand:
            if d in exclude:
                continue
            dset = self._doc_set(d)
            scores[d] = (len(q & dset) / len(q | dset), sorted(q & dset))
        top = heapq.nlargest(k, scores.items(), key=lambda x: x[1][0])
        return [{"doc": d, "score": s, "title_cos": 0, "body_cos": 0,
                 "g": self.idx.g[d], "matched": m} for d, (s, m) in top]


def idf_ok(idx, t, idf_min):
    return idx.idf(t) >= idf_min
