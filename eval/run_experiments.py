"""Runs all experiments and writes tables + plots to results/.

    python -m eval.run_experiments                 # dev tuning + test results
    python -m eval.run_experiments --max-test 300  # quicker run

E1 baseline (single search) vs multi-hop agent
E2 split by question type (bridge / comparison)
E3 ablations: title-zone weight, stemming, stop words   (tuned on DEV split)
E4 scoring function: lnc.ltc vs raw tf vs Jaccard
E5 fusion: chain score vs RRF, alpha values
E6 efficiency: full scoring vs index elimination vs champion lists (+ g(d))
E7 failure analysis of the agent on bridge questions
E8 "shortcut" questions the single search already solves
"""
import argparse
import json
import os
import time

from agent.pipeline import MultiHopAgent, baseline
from eval.metrics import both_at_k, evaluate, interpolated_pr
from ir.indexer import InvertedIndex
from ir.scorer import Scorer

COLS = ["R@2", "R@5", "R@10", "Both@2", "Both@5", "Both@10", "P@2", "MAP", "nDCG@10"]


def load_questions(path, idx):
    title2id = {d["title"]: i for i, d in enumerate(idx.docs)}
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            q["gold"] = [title2id[t] for t in q["gold_titles"] if t in title2id]
            if len(q["gold"]) == len(q["gold_titles"]):
                qs.append(q)
    return qs


def run(system, qs, k=10):
    t0 = time.perf_counter()
    rankings = [system(q["question"], k) for q in qs]
    ms = 1000 * (time.perf_counter() - t0) / max(len(qs), 1)
    m = evaluate(rankings, [q["gold"] for q in qs])
    m["ms/query"] = ms
    return m, rankings


def agent_fn(scorer, **kw):
    ag = MultiHopAgent(scorer, **kw)
    return lambda q, k: ag.search(q, k)["ranking"]


def base_fn(scorer):
    return lambda q, k: baseline(scorer, q, k)


def table(rows, title, extra=("ms/query",)):
    cols = COLS + list(extra)
    out = [f"\n### {title}\n", "| System | n | " + " | ".join(cols) + " |",
           "|---|---|" + "---|" * len(cols)]
    for name, n, m in rows:
        vals = [f"{m[c]:.1f}" if c == "ms/query" else f"{m[c]:.3f}" for c in cols]
        out.append(f"| {name} | {n} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def get_index(path, corpus, stem=True, stop=True):
    if not os.path.exists(path):
        print("building", path)
        InvertedIndex(stem=stem, stop=stop).build(corpus).save(path)
    return InvertedIndex.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-test", type=int, default=500)
    ap.add_argument("--max-dev", type=int, default=200)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    corpus = f"{a.data}/corpus.jsonl"
    idx = get_index(f"{a.data}/index.pkl", corpus)
    allq = load_questions(f"{a.data}/questions.jsonl", idx)
    dev = [q for q in allq if q["split"] == "dev"][:a.max_dev] or allq
    test = [q for q in allq if q["split"] == "test"][:a.max_test] or allq
    print(f"corpus: {idx.N} paragraphs | dev {len(dev)} | test {len(test)} questions")
    report = [f"# HopSearch results\n\nCorpus: {idx.N} paragraphs, {len(idx.df)} terms. "
              f"Dev questions: {len(dev)}. Test questions: {len(test)}.\n"]
    csv_rows = []

    def record(exp, name, qs, m):
        csv_rows.append({"experiment": exp, "system": name, "n": len(qs), **m})
        print(f"  {exp:4s} {name:40s} Both@10={m['Both@10']:.3f} R@10={m['R@10']:.3f} "
              f"{m['ms/query']:.1f} ms")

    # ---------------- E3a: tune title-zone weight on DEV (agent Both@10)
    print("E3 tuning title weight on dev ...")
    rows, best_w, best_v = [], 0.4, (-1,)
    for w in (0.0, 0.2, 0.4, 0.6, 0.8):
        s = Scorer(idx, w_title=w, w_body=1 - w)
        m, _ = run(agent_fn(s), dev)
        record("E3", f"agent w_title={w}", dev, m)
        rows.append((f"Agent, w_title={w}", len(dev), m))
        key = (m["Both@10"], m["Both@2"], m["MAP"])   # tie-break on top ranks
        if key > best_v:
            best_v, best_w = key, w
    report.append(table(rows, "E3a. Zone weighting (title vs body), DEV split"))
    report.append(f"\nChosen w_title = **{best_w}** (best Both@10 on dev).\n")

    # alpha tuning on dev
    rows, best_a, best_v = [], 0.5, (-1,)
    S = Scorer(idx, w_title=best_w, w_body=1 - best_w)
    for al in (0.3, 0.5, 0.7):
        m, _ = run(agent_fn(S, alpha=al), dev)
        record("E5", f"agent alpha={al}", dev, m)
        rows.append((f"Agent chain fusion, alpha={al}", len(dev), m))
        key = (m["Both@10"], m["Both@2"], m["MAP"])
        if key > best_v:
            best_v, best_a = key, al
    m, _ = run(agent_fn(S, alpha=best_a, fusion="rrf"), dev)
    record("E5", "agent RRF fusion", dev, m)
    rows.append(("Agent RRF fusion", len(dev), m))
    report.append(table(rows, "E5. Fusion of hop-1 and hop-2 results, DEV split"))
    report.append(f"\nChosen alpha = **{best_a}**.\n")

    # ---------------- E1 + E2 on TEST
    print("E1/E2 baseline vs agent on test ...")
    mb, rank_b = run(base_fn(S), test)
    ma, rank_a = run(agent_fn(S, alpha=best_a), test)
    record("E1", "baseline", test, mb)
    record("E1", "agent", test, ma)
    report.append(table([("Single-search baseline (tf-idf)", len(test), mb),
                         ("Multi-hop agent", len(test), ma)],
                        "E1. Main result: single search vs multi-hop agent, TEST split"))
    rows = []
    for typ in ("bridge", "comparison"):
        sub = [i for i, q in enumerate(test) if q["type"] == typ]
        if not sub:
            continue
        g = [test[i]["gold"] for i in sub]
        for name, rk in (("Baseline", rank_b), ("Agent", rank_a)):
            m = evaluate([rk[i] for i in sub], g)
            m["ms/query"] = 0
            rows.append((f"{name} - {typ}", len(sub), m))
    report.append(table(rows, "E2. By question type, TEST split", extra=()))

    # decomposer accuracy
    from agent.decompose import decompose
    acc = sum(decompose(q["question"])["type"] == q["type"] for q in test) / len(test)
    report.append(f"\nQuestion-type detector accuracy vs HotpotQA labels: **{acc:.3f}**\n")

    # PR curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        levels = [i / 10 for i in range(11)]
        plt.figure(figsize=(5, 4))
        for name, rk in (("Single-search baseline", rank_b), ("Multi-hop agent", rank_a)):
            curves = [interpolated_pr(r, q["gold"]) for r, q in zip(rk, test)]
            avg = [sum(c[i] for c in curves) / len(curves) for i in range(11)]
            plt.plot(levels, avg, marker="o", label=name)
        plt.xlabel("Recall"); plt.ylabel("Interpolated precision")
        plt.title("11-point interpolated P-R (top 10)"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("results/pr_curve.png", dpi=150)
        report.append("\n![P-R curve](pr_curve.png)\n")
    except ImportError:
        pass

    # ---------------- E3b: stemming / stop words (rebuild indexes)
    print("E3 stemming / stop-word ablations ...")
    rows = [("Full pipeline (stem + stop words removed)", len(test), ma)]
    for tag, stem, stop in (("nostem", False, True), ("nostop", True, False)):
        ix = get_index(f"{a.data}/index_{tag}.pkl", corpus, stem, stop)
        s2 = Scorer(ix, w_title=best_w, w_body=1 - best_w)
        tq = load_questions(f"{a.data}/questions.jsonl", ix)
        tq = [q for q in tq if q["split"] == "test"][:a.max_test] or tq
        m, _ = run(agent_fn(s2, alpha=best_a), tq)
        record("E3", f"agent {tag}", tq, m)
        rows.append((f"No {'stemming' if tag == 'nostem' else 'stop-word removal'}", len(tq), m))
        del ix
    report.append(table(rows, "E3b. Text-processing ablations (agent), TEST split"))

    # ---------------- E4 scoring functions (baseline, single search)
    print("E4 scoring functions ...")
    rows = []
    for meth in ("lnc.ltc", "tf", "jaccard"):
        s2 = Scorer(idx, w_title=best_w, w_body=1 - best_w, method=meth)
        m, _ = run(base_fn(s2), test)
        record("E4", f"baseline {meth}", test, m)
        rows.append((f"Single search, {meth}", len(test), m))
    report.append(table(rows, "E4. Scoring function, TEST split"))

    # ---------------- E6 efficiency
    print("E6 efficiency ...")
    rows = [("Agent, full cosine scoring", len(test), ma)]
    for name, kw in (("Index elimination (idf >= 1.0, >= 2 terms)", {"mode": "idf_elim", "idf_min": 1.0}),
                     ("Champion lists (r = 100)", {"mode": "champion"}),
                     ("Full + static quality g(d), lambda = 0.1", {"lam": 0.1})):
        s2 = Scorer(idx, w_title=best_w, w_body=1 - best_w, **kw)
        m, _ = run(agent_fn(s2, alpha=best_a), test)
        record("E6", name, test, m)
        rows.append((f"Agent, {name}", len(test), m))
    report.append(table(rows, "E6. Efficiency and net score, TEST split"))

    # ---------------- E7 failure analysis + E8 shortcuts
    print("E7/E8 analysis ...")
    ag = MultiHopAgent(S, alpha=best_a)
    cats, examples = {}, []
    bridge_q = [q for q in test if q["type"] == "bridge"]
    shortcut = 0
    for q in bridge_q:
        b = baseline(S, q["question"], 10)
        shortcut += both_at_k(b, q["gold"], 10)
        out = ag.search(q["question"], 10)
        if both_at_k(out["ranking"], q["gold"], 10):
            continue
        hop1 = [r["doc"] for r in out["trail"][0]["results"]]
        hop1_full = [r["doc"] for r in S.search(q["question"], 10)]
        bridges = next((t["bridges"] for t in out["trail"] if t["step"] == "bridges"), [])
        bdocs = {x["doc"] for x in bridges}
        found = [g for g in q["gold"] if g in hop1_full]
        if not found:
            c = "hop 1 found neither gold paragraph"
        elif not (set(q["gold"]) - set(hop1_full)) & bdocs:
            c = "hop 1 OK, but missing paragraph was not among bridge candidates"
        else:
            c = "bridge found, but ranked below top-10 after fusion"
        cats[c] = cats.get(c, 0) + 1
        if len(examples) < 25:
            examples.append(f"- **{q['question']}**  \n  gold: {q['gold_titles']}  \n  "
                            f"hop-1 top-3: {[idx.docs[d]['title'] for d in hop1[:3]]}  \n  "
                            f"bridges: {[x['entity'] for x in bridges]}  \n  category: {c}")
    report.append("\n### E7. Failure analysis (agent, bridge questions, Both@10 = 0)\n")
    report.append("| Failure type | Count |\n|---|---|")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        report.append(f"| {c} | {n} |")
    report.append("\nExamples:\n\n" + "\n".join(examples))
    if bridge_q:
        report.append(f"\n### E8. Shortcut questions\n\nThe single-search baseline already finds both "
                      f"gold paragraphs (Both@10) for **{int(shortcut)}/{len(bridge_q)} "
                      f"({shortcut/len(bridge_q):.1%})** of bridge questions, so these did not "
                      f"really need a second hop.\n")

    with open("results/results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    import csv
    with open("results/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    print("\nwrote results/results.md, results/results.csv, results/pr_curve.png")


if __name__ == "__main__":
    main()
