"""Prints the intermediate output of the pipeline for one question, step by step.
Made for the demo video: shows postings, weights, cosine scores, top-K, bridges, hop 2.

    python inspect_pipeline.py                      # uses your first demo question automatically
    python inspect_pipeline.py "Your question here?" # or type any question in quotes
"""
import json
import os
import math
import re
import sys
from collections import Counter

from agent.bridge import find_bridges
from agent.pipeline import MultiHopAgent
from ir.indexer import InvertedIndex
from ir.preprocess import STOP_WORDS, analyze, tokenize
from ir.scorer import Scorer


def line(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


if not os.path.exists("data/index.pkl"):
    sys.exit("No index found. Run this from the hopsearch folder after: python -m ir.indexer")


def default_question():
    """First question from demo_questions.txt (made by find_demo_questions.py),
    otherwise the first bridge question in data/questions.jsonl."""
    if os.path.exists("demo_questions.txt"):
        with open("demo_questions.txt", encoding="utf-8") as f:
            first = f.readline().strip()
        if first:
            return first.split("   [answer")[0].strip()
    with open("data/questions.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["type"] == "bridge":
                return q["question"]


arg = " ".join(sys.argv[1:]).strip()
q = arg if arg and not re.fullmatch(r"\[?Q\d*\]?", arg) else default_question()
idx = InvertedIndex.load("data/index.pkl")
s = Scorer(idx, w_title=0.2, w_body=0.8)
print(f"Collection: N = {idx.N:,} paragraphs, {len(idx.df):,} terms in the dictionary")
print(f"Question: {q}")

line("STEP 1  Text processing (tokenise -> case-fold -> stop words -> Porter stem)")
toks = tokenize(q)
print("tokens      :", toks)
print("stop words  :", [t for t in toks if t in STOP_WORDS])
terms = [t for t, _ in analyze(q)]
print("index terms :", terms)

line("STEP 2  Postings from the inverted index  (term -> df -> first postings)")
for t in terms:
    plist = idx.postings.get(t, {})
    sample = list(plist.items())[:3]
    shown = ", ".join(f"doc {d} [{idx.docs[d]['title'][:22]}] title{list(p[0])} body{list(p[1])[:4]}"
                      for d, p in sample)
    print(f"{t:12s} df={len(plist):5d}  idf=log10({idx.N}/{len(plist) or 1})={idx.idf(t):.3f}")
    print(f"{'':12s} -> {shown}")

line("STEP 3  Query vector, ltc  (1 + log10 tf) x idf, then cosine-normalised")
tf = Counter(t for t in terms if t in idx.df)
raw = {t: (1 + math.log10(c)) * idx.idf(t) for t, c in tf.items()}
norm = math.sqrt(sum(v * v for v in raw.values())) or 1
print(f"{'term':12s} {'tf':>3s} {'idf':>7s} {'tf.idf':>8s} {'normalised':>11s}")
for t in raw:
    print(f"{t:12s} {tf[t]:3d} {idx.idf(t):7.3f} {raw[t]:8.3f} {raw[t]/norm:11.3f}")
print(f"|q| = {norm:.3f}   -> rarest terms get the largest weight")

line("STEP 4  Ranked retrieval, lnc.ltc cosine, zones (0.2 title + 0.8 body), heap top-5")
res = s.search(q, 5)
for i, r in enumerate(res, 1):
    print(f"#{i} score={r['score']:.4f}  cos_title={r['title_cos']:.3f}  cos_body={r['body_cos']:.3f}"
          f"  [{idx.docs[r['doc']]['title']}]  matched={r['matched']}")
if res:
    d = res[0]["doc"]
    print(f"\nWorked cosine for #1 [{idx.docs[d]['title']}], body zone:")
    for t in res[0]["matched"]:
        tb = len(idx.postings[t][d][1])
        if tb:
            wd = (1 + math.log10(tb)) / idx.norm["body"][d]
            print(f"   {t:12s} w_q={raw[t]/norm:.3f}  x  w_d=(1+log10 {tb})/{idx.norm['body'][d]:.2f}={wd:.3f}"
                  f"  = {raw[t]/norm*wd:.4f}")

line("STEP 5  Bridge entities (capitalised names in top-3 paragraphs that are titles)")
bridges = find_bridges(idx, s.search(q, 10), q)
for b in bridges:
    print(f"\"{b['entity']}\" -> title [{idx.docs[b['doc']]['title']}]  match={b['match']}"
          f"  conf={b['conf']:.2f}  found in [{idx.docs[b['from_doc']]['title']}]")
if not bridges:
    print("no bridge found")

line("STEP 6  Full agent: hop 2 queries and final fused ranking")
out = MultiHopAgent(s, alpha=0.5).search(q, 5)
print("question type:", out["plan"]["type"])
for t in out["trail"]:
    if t["step"] == "hop2":
        print(f"hop 2 via \"{t['bridge']}\": query terms {t['query_terms']}  (residual {t['residual']})")
    if t["step"] == "subquery":
        print(f"sub-query for \"{t['entity']}\"")
print("\nFINAL top 5:")
for i, d in enumerate(out["ranking"], 1):
    print(f"  {i}. {idx.docs[d]['title']}")
