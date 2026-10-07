"""Boolean retrieval (Lecture 1) and phrase queries (Lecture 2).

- AND uses the linear merge of two sorted postings lists.
- Query optimisation: process terms in order of increasing df, so the
  intermediate result stays as small as possible.
- Phrase queries use the positional index: "christopher nolan" matches only if
  'nolan' appears at position p+1 of some 'christoph' at position p.
"""
from ir.preprocess import analyze


def postings_list(idx, term):
    return sorted(idx.postings.get(term, {}))


def intersect(p1, p2):
    """The textbook merge algorithm: walk both sorted lists once, O(x + y)."""
    i = j = 0
    out = []
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            out.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return out


def union(p1, p2):
    return sorted(set(p1) | set(p2))


def and_not(p1, p2):
    s = set(p2)
    return [d for d in p1 if d not in s]


def boolean_and(idx, terms):
    """AND of many terms, rarest term first (increasing document frequency)."""
    if not terms:
        return []
    terms = sorted(set(terms), key=lambda t: idx.df.get(t, 0))
    result = postings_list(idx, terms[0])
    for t in terms[1:]:
        if not result:
            break
        result = intersect(result, postings_list(idx, t))
    return result


def boolean_or(idx, terms):
    result = []
    for t in terms:
        result = union(result, postings_list(idx, t))
    return result


def phrase_query(idx, text):
    """Docs where the analysed words of `text` appear consecutively (in title or body).
    Positions were stored before stop-word removal, so we compare gaps between the
    original positions."""
    q = analyze(text, idx.stem, idx.stop)
    if not q:
        return []
    terms = [t for t, _ in q]
    offsets = [p - q[0][1] for _, p in q]
    candidates = boolean_and(idx, terms)
    hits = []
    for d in candidates:
        for zone in (0, 1):  # 0 = title, 1 = body
            starts = set(idx.postings[terms[0]][d][zone])
            for t, off in zip(terms[1:], offsets[1:]):
                pos = {p - off for p in idx.postings[t][d][zone]}
                starts &= pos
                if not starts:
                    break
            if starts:
                hits.append(d)
                break
    return hits


def parse_and_run(idx, query):
    """Tiny Boolean query parser: 'nolan AND director AND NOT batman',
    'a OR b', and quoted phrases "christopher nolan". Left to right."""
    import re
    tokens = re.findall(r'"[^"]+"|\S+', query)
    result, op = None, "AND"
    for tok in tokens:
        up = tok.upper()
        if up in ("AND", "OR", "NOT"):
            op = "AND NOT" if (up == "NOT" and op == "AND") else up
            continue
        if tok.startswith('"'):
            plist = phrase_query(idx, tok.strip('"'))
        else:
            ts = [t for t, _ in analyze(tok, idx.stem, idx.stop)]
            plist = postings_list(idx, ts[0]) if ts else []
        if result is None:
            result = plist
        elif op == "AND":
            result = intersect(result, plist)
        elif op == "OR":
            result = union(result, plist)
        elif op in ("AND NOT", "NOT"):
            result = and_not(result, plist)
        op = "AND"
    return result or []
