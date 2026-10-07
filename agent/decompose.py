"""Query parser / decomposer (Lecture 7: query parsers turn one user request into
several sub-queries whose scores are merged).

HotpotQA has two question types:
  bridge     : "Which university did the director of Interstellar study at?"
               -> hop 1 finds the bridge entity (Christopher Nolan), hop 2 uses it.
  comparison : "Were Scott Derrickson and Ed Wood of the same nationality?"
               -> two INDEPENDENT sub-queries, one per entity.

We use simple, explainable rules (no ML).
"""
import re

from ir.preprocess import capitalized_spans

COMPARE_WORDS = r"(first|earlier|later|older|younger|more|less|larger|smaller|bigger|" \
                r"longer|shorter|taller|higher|lower|greater|fewer|newer|oldest|youngest)"
AUX = r"^(are|were|is|was|do|does|did|have|has|can|could)\b"


def _entities(question):
    q = question.strip().rstrip("?")
    # "..., X or Y" pattern: take the part after the last comma
    if " or " in q:
        tail = q.split(",")[-1] if "," in q else q
        parts = [p.strip() for p in re.split(r"\bor\b", tail) if p.strip()]
        if len(parts) == 2:
            spans = [capitalized_spans(p) for p in parts]
            ents = [s[-1] if s else p for s, p in zip(spans, parts)]
            return ents
    return capitalized_spans(q)


def decompose(question):
    """Return a plan dict: {"type": "bridge"|"comparison", "entities": [...],
    "subqueries": [...]}."""
    ql = question.lower().strip()
    ents = _entities(question)
    is_cmp = False
    if " or " in ql and (re.search(COMPARE_WORDS, ql) or ql.startswith(("which", "who", "what"))):
        is_cmp = len(ents) >= 2
    elif re.match(AUX, ql) and (" both " in ql or " same " in ql or " and " in ql):
        is_cmp = len(ents) >= 2
    if is_cmp:
        ents = ents[:2]
        # attribute words = question minus the entity names (e.g. "nationality")
        attr = question
        for e in ents:
            attr = attr.replace(e, " ")
        return {"type": "comparison", "entities": ents,
                "subqueries": [f"{e} {attr}" for e in ents], "attribute": attr.strip()}
    return {"type": "bridge", "entities": ents, "subqueries": [question]}
