"""HopSearch demo UI. Pure Python standard library, nothing to install.

    python app.py            then open http://localhost:8000

Pages
  /          Search: ask a question, see the search path, agent vs baseline,
             and "under the hood" details (query processing, bridges, scores)
  /boolean   Boolean search over the inverted index (AND / OR / NOT / "phrase")
  /results   The evaluation tables from results/results.md + the P-R curve
"""
import html
import json
import os
import random
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, quote, urlparse

from agent.pipeline import MultiHopAgent, baseline
from ir.boolean import parse_and_run
from ir.indexer import InvertedIndex
from ir.preprocess import STOP_WORDS, _stem, analyze, tokenize
from ir.scorer import Scorer

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
print("Loading index ...")
IDX = InvertedIndex.load(f"{DATA}/index.pkl")
SCORER = Scorer(IDX, w_title=0.2, w_body=0.8)   # tuned on dev (E3a)
AGENT = MultiHopAgent(SCORER, alpha=0.5)
QUESTIONS = [json.loads(l) for l in open(f"{DATA}/questions.jsonl", encoding="utf-8")]
QMAP = {q["question"]: q for q in QUESTIONS}
TITLE2ID = {d["title"]: i for i, d in enumerate(IDX.docs)}
E = html.escape

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
CSS = """
:root{
  --bg:#f6f7f9; --surface:#ffffff; --surface-2:#f1f3f6; --fg:#14171c; --muted:#5f6672;
  --line:#e3e6eb; --accent:#4f46e5; --accent-soft:#eef0ff; --accent-fg:#ffffff;
  --good:#15803d; --good-soft:#e8f6ee; --bad:#b42318; --bad-soft:#fdecea;
  --mark:#fff3bf; --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.06);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0f1115; --surface:#171a20; --surface-2:#1e222a; --fg:#e8eaed; --muted:#9aa1ad;
  --line:#2a2f38; --accent:#8b85ff; --accent-soft:#25234a; --accent-fg:#0f1115;
  --good:#4ade80; --good-soft:#15291d; --bad:#f87171; --bad-soft:#2d1717;
  --mark:#4a3f0b; --shadow:none;}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent);text-decoration:none}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}

/* top bar */
.top{position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
.top-in{max-width:1120px;margin:0 auto;padding:12px 16px;display:flex;align-items:center;gap:20px}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;font-size:17px;color:var(--fg)}
.logo-mark{width:28px;height:28px;border-radius:8px;background:var(--accent);display:grid;place-items:center}
.nav{display:flex;gap:4px;margin-left:auto}
.nav a{padding:7px 12px;border-radius:8px;color:var(--muted);font-weight:500;font-size:14px}
.nav a.on{background:var(--surface);color:var(--fg);box-shadow:var(--shadow);border:1px solid var(--line)}
.nav a:hover{color:var(--fg)}

@media(max-width:480px){.nav a{padding:6px 8px}.top-in{gap:8px}.sbox .btn{padding:10px 12px}}
main{max-width:1120px;margin:0 auto;padding:28px 16px 80px}

/* hero */
.hero{text-align:center;padding:36px 0 8px}
.hero h1{font-size:34px;line-height:1.15;letter-spacing:-.02em;margin:0 0 10px}
.hero p{color:var(--muted);max-width:640px;margin:0 auto 26px;font-size:16px}
.stats{display:flex;justify-content:center;gap:28px;flex-wrap:wrap;margin:26px 0 6px;color:var(--muted);font-size:13px}
.stats b{display:block;color:var(--fg);font-size:20px;font-variant-numeric:tabular-nums}

/* search box */
.sbox{display:flex;gap:8px;background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:6px;box-shadow:var(--shadow);max-width:820px;margin:0 auto}
.sbox input{flex:1;min-width:0;border:0;outline:0;background:transparent;color:var(--fg);font-size:16px;padding:10px 12px}
.btn{border:1px solid var(--line);background:var(--surface);color:var(--fg);border-radius:10px;
  padding:10px 16px;font-weight:600;font-size:14px;cursor:pointer;white-space:nowrap;display:inline-flex;align-items:center;gap:6px}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-fg)}
.btn:hover{filter:brightness(.97)}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:860px;margin:16px auto 0}
.chip{background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:6px 12px;
  font-size:13px;color:var(--fg);max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip:hover{border-color:var(--accent)}

/* sections */
.section{margin-top:34px}
.section h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 12px;font-weight:700}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow)}

/* verdict */
.verdict{display:grid;grid-template-columns:1fr auto auto;gap:14px;align-items:center}
.q{font-size:20px;font-weight:650;letter-spacing:-.01em;line-height:1.35}
.meta{color:var(--muted);font-size:13px;margin-top:6px}
.meta b{color:var(--fg)}
.score{border-radius:12px;padding:10px 14px;text-align:center;min-width:120px;border:1px solid var(--line)}
.score .n{font-size:22px;font-weight:750;font-variant-numeric:tabular-nums}
.score .l{font-size:12px;color:var(--muted)}
.score.good{background:var(--good-soft);border-color:transparent}.score.good .n{color:var(--good)}
.score.bad{background:var(--bad-soft);border-color:transparent}.score.bad .n{color:var(--bad)}
@media(max-width:720px){.verdict{grid-template-columns:1fr 1fr}.verdict>div:first-child{grid-column:1/-1}}

/* search path */
.path{display:grid;grid-template-columns:1fr 40px 220px 40px 1fr;align-items:stretch}
.path.cmp{grid-template-columns:1fr 40px 1fr}
.arrow{display:grid;place-items:center;color:var(--muted)}
.node{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:var(--shadow);min-width:0}
.node .step{font-size:12px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
.node .title{font-weight:650;margin:4px 0 6px;font-size:16px}
.node .text{font-size:13.5px;color:var(--muted);display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden}
.node.gold{border-color:var(--good);box-shadow:0 0 0 3px var(--good-soft)}
.bridge{background:var(--accent-soft);border:1px dashed var(--accent);border-radius:14px;padding:14px;
  display:flex;flex-direction:column;justify-content:center;text-align:center}
.bridge .ent{font-size:17px;font-weight:700;color:var(--fg);margin:6px 0}
.bridge .how{font-size:12px;color:var(--muted)}
.qterms{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.term{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;background:var(--surface-2);
  border:1px solid var(--line);border-radius:6px;padding:1px 7px}
.term.rq{border-color:var(--accent);color:var(--accent)}
@media(max-width:860px){.path,.path.cmp{grid-template-columns:1fr}.arrow{height:30px;transform:rotate(90deg)}}

/* side by side */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:860px){.cols{grid-template-columns:1fr}}
.colhead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
.colhead h3{margin:0;font-size:16px}
.colhead span{font-size:13px;color:var(--muted)}
.res{display:flex;gap:12px;padding:12px 0;border-top:1px solid var(--line)}
.res:first-of-type{border-top:0}
.rank{flex:none;width:26px;height:26px;border-radius:8px;background:var(--surface-2);display:grid;place-items:center;
  font-size:13px;font-weight:700;color:var(--muted)}
.res.gold .rank{background:var(--good);color:#fff}
.res .t{font-weight:600}
.res .x{font-size:13px;color:var(--muted);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;margin-left:6px;vertical-align:2px}
.badge.gold{background:var(--good-soft);color:var(--good)}
mark{background:var(--mark);color:inherit;border-radius:3px;padding:0 1px}

/* under the hood */
details{background:var(--surface);border:1px solid var(--line);border-radius:14px;margin-top:10px;box-shadow:var(--shadow)}
summary{cursor:pointer;padding:14px 18px;font-weight:600;list-style:none;display:flex;align-items:baseline;gap:10px}
summary::-webkit-details-marker{display:none}
summary::after{margin-left:auto;content:"+";color:var(--muted);font-weight:400;font-size:18px;line-height:1}
details[open] summary::after{content:"\\2212"}
summary small{color:var(--muted);font-weight:400}
.dbody{padding:0 18px 16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th{color:var(--muted);font-weight:600;text-align:left;padding:8px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:8px;border-bottom:1px solid var(--line);vertical-align:top}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.goldrow td{background:var(--good-soft)}
.pipe{display:grid;grid-template-columns:150px 1fr;gap:8px 14px;align-items:start;font-size:13px}
.pipe .lbl{color:var(--muted)}
.strike{text-decoration:line-through;opacity:.5}
.note{font-size:13px;color:var(--muted);margin:8px 0 0}

/* boolean + results */
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:760px){.grid3{grid-template-columns:1fr}}
.kpi .n{font-size:28px;font-weight:750;font-variant-numeric:tabular-nums}
.kpi .l{color:var(--muted);font-size:13px}
.kpi .d{font-size:13px;color:var(--good);font-weight:600}
.md h3{font-size:17px;margin:30px 0 10px}
.md p{color:var(--muted)}
.md table{background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.md .tw{overflow-x:auto}
img.curve{max-width:520px;width:100%;border-radius:12px;border:1px solid var(--line);background:#fff}
.empty{text-align:center;color:var(--muted);padding:40px}
"""

LOGO = ('<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="12" r="2.5"/><circle cx="19" cy="6" r="2.5"/>'
        '<circle cx="19" cy="18" r="2.5"/><path d="M7.3 11 16.7 7M7.3 13l9.4 4"/></svg>')
ARROW = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
         'stroke-linecap="round"><path d="M4 12h15M13 6l6 6-6 6"/></svg>')


def layout(body, active="search"):
    def tab(href, name, key):
        return f'<a href="{href}" class="{"on" if active == key else ""}">{name}</a>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HopSearch</title>
<style>{CSS}</style></head><body>
<header class="top"><div class="top-in">
<a class="logo" href="/"><span class="logo-mark">{LOGO}</span>HopSearch</a>
<nav class="nav">{tab("/", "Search", "search")}{tab("/boolean", "Boolean", "boolean")}{tab("/results", "Results", "results")}</nav>
</div></header><main>{body}</main></body></html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def highlight(text, qterms):
    def rep(m):
        w = m.group(0)
        toks = tokenize(w)
        return f"<mark>{E(w)}</mark>" if toks and _stem(toks[0]) in qterms else E(w)
    return re.sub(r"[A-Za-z0-9À-ɏ']+", rep, text)


def search_box(q=""):
    return f"""<form class="sbox" action="/search">
<input name="q" value="{E(q)}" placeholder="Ask a two-step question…" autocomplete="off" autofocus>
<a class="btn" href="/random" title="Pick a random HotpotQA question">Random</a>
<button class="btn primary">Search</button></form>"""


def result_list(ranking, gold, qterms, n=5):
    rows = []
    for i, d in enumerate(ranking[:n], 1):
        doc = IDX.docs[d]
        g = d in gold
        rows.append(f'<div class="res{" gold" if g else ""}"><div class="rank">{i}</div><div>'
                    f'<div class="t">{E(doc["title"])}{"<span class=\"badge gold\">evidence</span>" if g else ""}</div>'
                    f'<div class="x">{highlight(doc["text"], qterms)}</div></div></div>')
    return "".join(rows) or '<p class="note">No results.</p>'


def score_table(results, gold):
    rows = "".join(
        f'<tr class="{"goldrow" if r["doc"] in gold else ""}"><td class="num">{i}</td>'
        f'<td>{E(IDX.docs[r["doc"]]["title"])}</td>'
        f'<td class="num">{r["title_cos"]:.3f}</td><td class="num">{r["body_cos"]:.3f}</td>'
        f'<td class="num">{r["g"]:.2f}</td><td class="num"><b>{r["score"]:.3f}</b></td>'
        f'<td class="mono">{E(", ".join(r["matched"]))}</td></tr>'
        for i, r in enumerate(results, 1))
    return ('<table><tr><th class="num">#</th><th>Paragraph</th><th class="num">cos(title)</th>'
            '<th class="num">cos(body)</th><th class="num">g(d)</th><th class="num">score</th>'
            f'<th>matched terms</th></tr>{rows}</table>'
            '<p class="note">score = 0.2 · cos(title) + 0.8 · cos(body), lnc.ltc weighting, top-K by heap.</p>')


def terms_html(ts, residual=()):
    return '<div class="qterms">' + "".join(
        f'<span class="term{" rq" if t in residual else ""}">{E(t)}</span>' for t in ts) + "</div>"


def node(step, doc_id, gold, qterms):
    doc = IDX.docs[doc_id]
    g = doc_id in gold
    return (f'<div class="node{" gold" if g else ""}"><div class="step">{step}</div>'
            f'<div class="title">{E(doc["title"])}{"<span class=\"badge gold\">evidence</span>" if g else ""}</div>'
            f'<div class="text">{highlight(doc["text"], qterms)}</div></div>')


def processing_panel(q):
    toks = tokenize(q)
    kept = [(t, s) for (s, _), t in zip(analyze(q), [t for t in toks if t not in STOP_WORDS])]
    tok_html = " ".join(f'<span class="term{" strike" if t in STOP_WORDS else ""}">{E(t)}</span>' for t in toks)
    stem_html = " ".join(f'<span class="term">{E(t)} → {E(s)}</span>' if t != s else
                         f'<span class="term">{E(s)}</span>' for t, s in kept)
    idf_html = " ".join(f'<span class="term">{E(s)} · df {IDX.df.get(s, 0)} · idf {IDX.idf(s):.2f}</span>'
                        for _, s in kept)
    return f"""<div class="pipe">
<div class="lbl">1. Tokenise + case-fold</div><div>{tok_html}</div>
<div class="lbl">2. Remove stop words</div><div class="note" style="margin:0">crossed-out tokens are stop words</div>
<div class="lbl">3. Porter stemming</div><div>{stem_html}</div>
<div class="lbl">4. Index statistics</div><div>{idf_html}</div></div>
<p class="note">Rare terms (high idf) dominate the query vector; common ones barely count.</p>"""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def home_page():
    examples = [q for q in QUESTIONS if q["type"] == "bridge" and len(q["question"]) < 90][:4] + \
               [q for q in QUESTIONS if q["type"] == "comparison" and len(q["question"]) < 90][:2]
    chips = "".join(f'<a class="chip" href="/search?q={quote(q["question"])}">{E(q["question"])}</a>'
                    for q in examples)
    body = f"""<section class="hero">
<h1>Search that takes two steps</h1>
<p>Some questions need two searches: the first finds a linking name, the second uses it.
HopSearch does this automatically and shows every step it took.</p>
{search_box()}
<div class="chips">{chips}</div>
<div class="stats"><div><b>{IDX.N:,}</b>Wikipedia paragraphs</div><div><b>{len(IDX.df):,}</b>index terms</div>
<div><b>{len(QUESTIONS):,}</b>HotpotQA questions</div></div></section>
<section class="section"><h2>How it works</h2><div class="grid3">
<div class="card"><div class="step mono" style="color:var(--accent);font-weight:700">01 · HOP 1</div>
<p style="margin:6px 0 0">Rank all paragraphs for the question with tf-idf cosine (lnc.ltc), title and body zones, heap top-K.</p></div>
<div class="card"><div class="step mono" style="color:var(--accent);font-weight:700">02 · BRIDGE</div>
<p style="margin:6px 0 0">Find names in the top paragraphs that are titles of other paragraphs: a dictionary lookup with edit distance.</p></div>
<div class="card"><div class="step mono" style="color:var(--accent);font-weight:700">03 · HOP 2</div>
<p style="margin:6px 0 0">Search again with the bridge name plus the unanswered question words, then fuse both hops.</p></div>
</div></section>"""
    return layout(body, "search")


def search_page(q):
    info = QMAP.get(q)
    gold = {TITLE2ID[t] for t in info["gold_titles"] if t in TITLE2ID} if info else set()
    out = AGENT.search(q, 10)
    base = baseline(SCORER, q, 10)
    qterms = set(SCORER.query_terms(q))
    plan = out["plan"]
    trail = {t["step"]: t for t in out["trail"]}
    hop2s = [t for t in out["trail"] if t["step"] == "hop2"]
    subs = [t for t in out["trail"] if t["step"] == "subquery"]

    # ---- verdict
    if info:
        a2 = sum(d in gold for d in out["ranking"][:2])
        b2 = sum(d in gold for d in base[:2])
        meta = (f'<div class="meta">Dataset answer: <b>{E(info["answer"])}</b> · evidence paragraphs: '
                f'<b>{E(" + ".join(info["gold_titles"]))}</b></div>')
        scores = (f'<div class="score {"good" if a2 == 2 else "bad"}"><div class="n">{a2}/2</div>'
                  f'<div class="l">HopSearch top 2</div></div>'
                  f'<div class="score {"good" if b2 == 2 else "bad"}"><div class="n">{b2}/2</div>'
                  f'<div class="l">Single search top 2</div></div>')
    else:
        meta = '<div class="meta">Your own question: no gold labels, so no score.</div>'
        scores = ""
    kind = "Bridge question · two chained searches" if plan["type"] == "bridge" else \
           "Comparison question · one search per entity"
    verdict = (f'<div class="card verdict"><div><div class="meta" style="margin:0 0 4px">{kind}</div>'
               f'<div class="q">{E(q)}</div>{meta}</div>{scores}</div>')

    # ---- search path
    if plan["type"] == "bridge":
        chains = trail.get("fusion", {}).get("chains", [])
        if chains:
            best = chains[0]
            h2 = next((h for h in hop2s if h["bridge"] == best["bridge"]), None)
            br = next((b for b in trail["bridges"]["bridges"] if b["entity"] == best["bridge"]), None)
            how = f'{br["match"].replace("_", " ")} title match · conf {br["conf"]:.2f}' if br else ""
            q2 = terms_html(h2["query_terms"], h2["residual"]) if h2 else ""
            others = [b["entity"] for b in trail["bridges"]["bridges"] if b["entity"] != best["bridge"]]
            path = (f'<div class="path">{node("Hop 1 · full question", best["p1"], gold, qterms)}'
                    f'<div class="arrow">{ARROW}</div>'
                    f'<div class="bridge"><div class="step mono" style="color:var(--accent);font-weight:700;font-size:12px">BRIDGE ENTITY</div>'
                    f'<div class="ent">{E(best["bridge"])}</div><div class="how">{E(how)}</div>{q2}'
                    f'<div class="how" style="margin-top:8px">hop-2 query · <span style="color:var(--accent)">blue</span> = leftover question words</div></div>'
                    f'<div class="arrow">{ARROW}</div>{node("Hop 2 · via bridge", best["p2"], gold, qterms)}</div>')
            if others:
                path += f'<p class="note">Other bridge candidates considered: {E(", ".join(others))}</p>'
        else:
            path = ('<div class="card">No bridge entity found in the hop-1 paragraphs, so the ranking '
                    'falls back to hop 1 only.</div>')
    else:
        cards = []
        for t in subs:
            if t["results"]:
                cards.append(node(f'Sub-query · {E(t["entity"])}', t["results"][0]["doc"], gold, qterms))
        path = f'<div class="path cmp">{"<div class=arrow style=font-size:22px;font-weight:700>+</div>".join(cards)}</div>' \
               '<p class="note">Each entity is searched separately, then the lists are merged with Reciprocal Rank Fusion.</p>'

    # ---- side by side
    compare = f"""<div class="cols">
<div class="card"><div class="colhead"><h3>HopSearch agent</h3><span>multi-hop</span></div>
{result_list(out["ranking"], gold, qterms)}</div>
<div class="card"><div class="colhead"><h3>Single search</h3><span>tf-idf baseline</span></div>
{result_list(base, gold, qterms)}</div></div>"""

    # ---- under the hood
    hood = [f'<details><summary>Query processing <small>tokenise → stop words → stem → idf</small></summary>'
            f'<div class="dbody">{processing_panel(q)}</div></details>']
    if "hop1" in trail:
        hood.append(f'<details><summary>Hop 1 scores <small>top 5 for the full question</small></summary>'
                    f'<div class="dbody">{score_table(trail["hop1"]["results"], gold)}</div></details>')
    if "bridges" in trail and trail["bridges"]["bridges"]:
        rows = "".join(f'<tr><td><b>{E(b["entity"])}</b></td><td>{E(IDX.docs[b["doc"]]["title"])}</td>'
                       f'<td>{E(b["match"].replace("_", " "))}</td><td class="num">{b["conf"]:.2f}</td>'
                       f'<td>{E(IDX.docs[b["from_doc"]]["title"])}</td></tr>' for b in trail["bridges"]["bridges"])
        hood.append('<details><summary>Bridge candidates <small>dictionary lookup in the title index</small></summary>'
                    '<div class="dbody"><table><tr><th>Name in text</th><th>Matched title</th><th>Match type</th>'
                    f'<th class="num">Confidence</th><th>Found in</th></tr>{rows}</table>'
                    '<p class="note">exact 1.0 · first+last word 0.9 · sub-phrase 0.8 · edit distance ≤ 2 0.7, '
                    '× the source paragraph\'s relative hop-1 score.</p></div></details>')
    for h in hop2s:
        hood.append(f'<details><summary>Hop 2 via “{E(h["bridge_title"])}” <small>{len(h["query_terms"])} query terms</small></summary>'
                    f'<div class="dbody">{terms_html(h["query_terms"], h["residual"])}<br>{score_table(h["results"], gold)}</div></details>')
    for t in subs:
        hood.append(f'<details><summary>Sub-query: {E(t["entity"])}</summary>'
                    f'<div class="dbody">{score_table(t["results"], gold)}</div></details>')
    if trail.get("fusion", {}).get("chains"):
        rows = "".join(f'<tr><td class="num">{c["score"]:.3f}</td><td>{E(IDX.docs[c["p1"]]["title"])}</td>'
                       f'<td>{E(IDX.docs[c["p2"]]["title"])}</td><td>{E(c["bridge"])}</td></tr>'
                       for c in trail["fusion"]["chains"])
        hood.append('<details><summary>Fusion <small>chain score = 0.5 · s₁ + 0.5 · s₂ · conf</small></summary>'
                    '<div class="dbody"><table><tr><th class="num">Chain score</th><th>Hop 1</th><th>Hop 2</th>'
                    f'<th>Bridge</th></tr>{rows}</table></div></details>')

    body = (f'<div style="margin-bottom:22px">{search_box(q)}</div>{verdict}'
            f'<section class="section"><h2>Search path</h2>{path}</section>'
            f'<section class="section"><h2>Agent vs single search · top 5</h2>{compare}</section>'
            f'<section class="section"><h2>Under the hood</h2>{"".join(hood)}</section>')
    return layout(body, "search")


def boolean_page(bq):
    form = f"""<form class="sbox" action="/boolean" style="margin:0 0 10px">
<input name="b" value="{E(bq)}" placeholder='e.g. "christopher nolan" AND director AND NOT batman' autocomplete="off" autofocus>
<button class="btn primary">Run</button></form>
<p class="note" style="text-align:center">Operators: <code>AND</code> <code>OR</code> <code>NOT</code> and
<code>"quoted phrases"</code> (positional index). Plain words are combined with AND.</p>"""
    if not bq:
        body = (f'<section class="hero" style="padding-top:20px"><h1>Boolean retrieval</h1>'
                f'<p>Exact-match search straight on the inverted index: postings lists are merged term by term, '
                f'rarest term first.</p>{form}</section>')
        return layout(body, "boolean")
    docs = parse_and_run(IDX, bq)
    words = [w for w in re.findall(r'"[^"]+"|\S+', bq) if w.upper() not in ("AND", "OR", "NOT")]
    rows = []
    for w in words:
        for t, _ in analyze(w.strip('"')):
            rows.append((w, t, IDX.df.get(t, 0)))
    order = sorted({r[1] for r in rows}, key=lambda t: IDX.df.get(t, 0))
    trow = "".join(f'<tr><td>{E(w)}</td><td class="mono">{E(t)}</td><td class="num">{df:,}</td></tr>' for w, t, df in rows)
    stats = (f'<div class="card"><table><tr><th>Query word</th><th>Index term</th><th class="num">Postings (df)</th></tr>'
             f'{trow}</table><p class="note">Processing order for AND (smallest postings first): '
             f'{" → ".join(E(t) for t in order)}</p></div>')
    res = "".join(f'<div class="res"><div class="rank">{i}</div><div><div class="t">{E(IDX.docs[d]["title"])}</div>'
                  f'<div class="x">{E(IDX.docs[d]["text"])}</div></div></div>' for i, d in enumerate(docs[:30], 1))
    body = (f'{form}<section class="section"><h2>Postings</h2>{stats}</section>'
            f'<section class="section"><h2>{len(docs):,} matching paragraphs'
            f'{" · showing 30" if len(docs) > 30 else ""}</h2><div class="card">{res or "<p class=note>No matches.</p>"}</div></section>')
    return layout(body, "boolean")


def md_to_html(md):
    """Small Markdown renderer for results.md (headings, tables, paragraphs, lists)."""
    out, lines, i = [], md.splitlines(), 0
    inline = lambda s: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", E(s))
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in block if not re.match(r"^\|[-| ]+\|$", r)]
            head = "".join(f'<th class="{"num" if j else ""}">{inline(c)}</th>' for j, c in enumerate(cells[0]))
            body = "".join("<tr>" + "".join(f'<td class="{"num" if j and re.match(r"^[\d.]+$", c) else ""}">{inline(c)}</td>'
                                            for j, c in enumerate(r)) + "</tr>" for r in cells[1:])
            out.append(f'<div class="tw"><table><tr>{head}</tr>{body}</table></div>')
            continue
        if ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:])}</h3>")
        elif ln.startswith("# "):
            pass
        elif ln.startswith("!["):
            if os.path.exists("results/pr_curve.png"):
                out.append('<p><img class="curve" src="/pr_curve.png" alt="Precision-recall curve"></p>')
        elif ln.startswith(("- ", "  ", "Examples")):
            pass   # skip the long failure examples list
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def results_page():
    path = "results/results.md"
    if not os.path.exists(path):
        return layout('<div class="card empty">No results yet. Run <code>python -m eval.run_experiments</code> first.</div>',
                      "results")
    md = open(path, encoding="utf-8").read()
    kpi = ""
    m_b = re.search(r"\| Single-search baseline.*?\|\s*\d+\s*\|(.*)\|", md)
    m_a = re.search(r"\| Multi-hop agent\s*\|\s*\d+\s*\|(.*)\|", md)
    if m_a and m_b:
        b = [float(x) for x in m_b.group(1).split("|")]
        a = [float(x) for x in m_a.group(1).split("|")]
        # columns: R@2 R@5 R@10 Both@2 Both@5 Both@10 P@2 MAP nDCG@10 ms
        def k(label, j, sub):
            return (f'<div class="card kpi"><div class="l">{label}</div><div class="n">{a[j]:.1%}</div>'
                    f'<div class="d">+{(a[j]-b[j])*100:.1f} pts vs single search ({b[j]:.1%})</div>'
                    f'<div class="l">{sub}</div></div>')
        kpi = ('<div class="grid3">' + k("Both evidence paragraphs in top 2", 3, "Both@2") +
               k("Both evidence paragraphs in top 10", 5, "Both@10") + k("Mean average precision", 7, "MAP") + "</div>")
    body = (f'<section class="hero" style="padding-top:12px"><h1>Evaluation</h1>'
            f'<p>Held-out HotpotQA test questions. Relevant = the two gold evidence paragraphs. '
            f'Settings were tuned on a separate dev split.</p></section>{kpi}'
            f'<div class="md">{md_to_html(md)}</div>')
    return layout(body, "results")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def send(self, body, code=200, loc=None, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        if loc:
            self.send_header("Location", loc)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        p = parse_qs(u.query)
        try:
            if u.path == "/random":
                return self.send("", 302, "/search?q=" + quote(random.choice(QUESTIONS)["question"]))
            if u.path == "/search" and p.get("q", [""])[0].strip():
                return self.send(search_page(p["q"][0].strip()))
            if u.path == "/boolean":
                return self.send(boolean_page(p.get("b", [""])[0].strip()))
            if u.path == "/results":
                return self.send(results_page())
            if u.path == "/pr_curve.png" and os.path.exists("results/pr_curve.png"):
                return self.send(open("results/pr_curve.png", "rb").read(), ctype="image/png")
            return self.send(home_page())
        except Exception as e:  # keep the demo alive
            return self.send(layout(f'<div class="card">Something went wrong: {E(repr(e))}</div>'), 500)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = 8000
    print(f"HopSearch running at http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
