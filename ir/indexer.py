"""Positional, zoned inverted index (Lectures 1, 2, 7).

Document unit (Lecture 2): ONE Wikipedia paragraph = one document, with 2 zones:
    title  -> e.g. "Christopher Nolan"
    body   -> the paragraph text

Index layout (dictionary + postings, Lecture 1):
    postings[term] = {doc_id: (title_positions, body_positions)}   # positional index
    df[term]       = number of docs containing term                # document frequency

Precomputed at build time (Lecture 6/7):
    norm[zone][doc] = sqrt(sum_t (1 + log10 tf)^2)   -> lnc length normalisation
    champions[term] = top-r docs by weight            -> champion lists
    g[doc]          = static quality (in-link count)  -> net score
    title_dict      = normalised title -> doc ids     -> dictionary lookup (Lecture 3)
"""
import json
import math
import pickle
import re
import time
from collections import defaultdict

from ir.preprocess import analyze, capitalized_spans

ZONES = ("title", "body")


def norm_title(t):
    """Normalise a title for dictionary lookup: drop '(film)' etc., case-fold."""
    t = re.sub(r"\s*\(.*?\)\s*", " ", t)
    t = t.replace("’", "'")
    return re.sub(r"\s+", " ", t).strip().lower()


class InvertedIndex:
    def __init__(self, stem=True, stop=True):
        self.stem, self.stop = stem, stop
        self.docs = []                  # doc_id -> {"title", "text"}
        self.postings = {}
        self.df = {}
        self.norm = {z: [] for z in ZONES}
        self.title_dict = defaultdict(list)
        self.first_last = defaultdict(list)   # ("christopher","nolan") -> docs
        self.surname = defaultdict(list)       # "nolan" -> docs
        self.champions = {}
        self.g = []

    # ------------------------------------------------------------------ build
    def build(self, corpus_path, champion_r=100, verbose=True):
        t0 = time.time()
        post = defaultdict(dict)
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                doc_id = len(self.docs)
                self.docs.append({"title": d["title"], "text": d["text"]})
                zone_pos = {}
                for z, text in (("title", d["title"]), ("body", d["text"])):
                    tf = defaultdict(list)
                    for term, pos in analyze(text, self.stem, self.stop):
                        tf[term].append(pos)
                    zone_pos[z] = tf
                    self.norm[z].append(math.sqrt(sum((1 + math.log10(len(p))) ** 2
                                                      for p in tf.values())) or 1.0)
                for term in set(zone_pos["title"]) | set(zone_pos["body"]):
                    post[term][doc_id] = (tuple(zone_pos["title"].get(term, ())),
                                          tuple(zone_pos["body"].get(term, ())))
                nt = norm_title(d["title"])
                self.title_dict[nt].append(doc_id)
                words = nt.split()
                if len(words) >= 2:
                    self.first_last[(words[0], words[-1])].append(doc_id)
                    self.surname[words[-1]].append(doc_id)
        self.postings = dict(post)
        self.df = {t: len(p) for t, p in self.postings.items()}
        self.N = len(self.docs)
        self._build_champions(champion_r)
        self._build_static_quality()
        if verbose:
            print(f"Indexed {self.N} docs, {len(self.df)} terms in {time.time()-t0:.1f}s")
        return self

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log10(self.N / df) if df else 0.0

    def doc_weight(self, term, doc_id, zone):
        """lnc document weight: (1 + log10 tf) / norm  (no idf on the doc side)."""
        tp, bp = self.postings[term][doc_id]
        tf = len(tp) if zone == "title" else len(bp)
        return (1 + math.log10(tf)) / self.norm[zone][doc_id] if tf else 0.0

    def _build_champions(self, r):
        """Champion list (Lecture 7): for each term keep the r docs with the highest
        weight; at query time we only score docs in the champion lists."""
        for term, plist in self.postings.items():
            if len(plist) <= r:
                self.champions[term] = list(plist)
            else:
                w = [(self.doc_weight(term, d, "title") + self.doc_weight(term, d, "body"), d)
                     for d in plist]
                w.sort(reverse=True)
                self.champions[term] = [d for _, d in w[:r]]

    def _build_static_quality(self):
        """g(d) (Lecture 7): a query-independent 'authority' score. Here: how many
        other paragraphs mention this paragraph's title (like an in-link count).
        Normalised to [0, 1] with log scaling."""
        inlinks = [0] * self.N
        for doc_id, d in enumerate(self.docs):
            seen = set()
            for span in capitalized_spans(d["text"]):
                for target in self.title_dict.get(norm_title(span), ()):
                    if target != doc_id and target not in seen:
                        seen.add(target)
                        inlinks[target] += 1
        mx = max(inlinks) if inlinks else 0
        self.g = [math.log1p(c) / math.log1p(mx) if mx else 0.0 for c in inlinks]

    # -------------------------------------------------------------- persist
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        idx = cls()
        with open(path, "rb") as f:
            idx.__dict__.update(pickle.load(f))
        return idx


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the inverted index")
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--out", default="data/index.pkl")
    ap.add_argument("--no-stem", action="store_true")
    ap.add_argument("--no-stop", action="store_true")
    a = ap.parse_args()
    idx = InvertedIndex(stem=not a.no_stem, stop=not a.no_stop).build(a.corpus)
    idx.save(a.out)
    print("saved", a.out)
