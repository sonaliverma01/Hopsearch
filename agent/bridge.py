"""Bridge-entity extraction (Lecture 3: dictionaries, edit distance).

From a hop-1 paragraph we collect capitalised phrases ("Christopher Edward Nolan",
"University College London") and keep only those that name ANOTHER paragraph in
our collection. That check is a dictionary lookup in `idx.title_dict`
(a hash table: normalised title -> doc ids).

Match types, from most to least confident:
    exact       the phrase IS a title                          conf 1.0
    first_last  first + last word match a title                conf 0.9
                ("Christopher Edward Nolan" -> "Christopher Nolan")
    subspan     a sub-phrase is a title                        conf 0.8
    fuzzy       Levenshtein distance <= 2 to a title           conf 0.7
                (handles spelling variants, Lecture 3)
"""
from ir.indexer import norm_title
from ir.preprocess import capitalized_spans, terms

CONF = {"exact": 1.0, "first_last": 0.9, "subspan": 0.8, "fuzzy": 0.7}


def edit_distance(a, b):
    """Levenshtein distance by dynamic programming (Lecture 3)."""
    if abs(len(a) - len(b)) > 2:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fuzzy_index(idx):
    """Bucket titles by first 2 letters so fuzzy matching only compares a few."""
    if not hasattr(idx, "_fuzzy"):
        b = {}
        for t in idx.title_dict:
            if len(t) >= 6:
                b.setdefault(t[:2], []).append(t)
        idx._fuzzy = b
    return idx._fuzzy


def lookup(idx, span):
    """Return list of (doc_id, match_type) for a phrase."""
    nt = norm_title(span)
    if nt in idx.title_dict:
        return [(d, "exact") for d in idx.title_dict[nt]]
    words = nt.split()
    if len(words) >= 3 and (words[0], words[-1]) in idx.first_last:
        return [(d, "first_last") for d in idx.first_last[(words[0], words[-1])]]
    for n in range(len(words) - 1, 0, -1):          # longest sub-phrase first
        for i in range(len(words) - n + 1):
            sub = " ".join(words[i:i + n])
            if len(sub) > 3 and sub in idx.title_dict:
                return [(d, "subspan") for d in idx.title_dict[sub]]
    if len(nt) >= 6:
        best = [t for t in _fuzzy_index(idx).get(nt[:2], []) if edit_distance(nt, t) <= 2]
        if best:
            return [(d, "fuzzy") for t in best[:2] for d in idx.title_dict[t]]
    return []


def find_bridges(idx, hop1_results, question, top_p=3, max_bridges=3):
    """Look at the top-p hop-1 paragraphs and return bridge candidates:
    [{"doc", "entity", "match", "conf", "from_doc"}], best first."""
    q_terms = set(terms(question, idx.stem, idx.stop))
    best = {}
    top_score = hop1_results[0]["score"] if hop1_results else 1.0
    for r in hop1_results[:top_p]:
        src = r["doc"]
        para = idx.docs[src]
        for span in capitalized_spans(para["text"]):
            span_terms = set(terms(span, idx.stem, idx.stop))
            if not span_terms or span_terms <= q_terms:
                continue          # already in the question -> not a NEW entity
            for d, mtype in lookup(idx, span):
                if d == src:
                    continue
                conf = CONF[mtype] * (r["score"] / top_score)
                if d not in best or conf > best[d]["conf"]:
                    best[d] = {"doc": d, "entity": span, "match": mtype,
                               "conf": conf, "from_doc": src}
    return sorted(best.values(), key=lambda b: -b["conf"])[:max_bridges]
