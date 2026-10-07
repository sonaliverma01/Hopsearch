"""Text processing pipeline (Lecture 2: tokenization, normalization, case folding,
stop words, stemming).

The SAME function is used for documents and queries, so that a query term and a
document term end up as the same index term (equivalence classing).
"""
import re
from functools import lru_cache

from ir.porter import stem as porter_stem

# Small English stop list + question words that carry no topical meaning.
# NOTE: we deliberately keep words like "first", "same", "born" (they carry meaning
# in HotpotQA questions) and numbers (years matter: "1995").
STOP_WORDS = set("""
a an the and or but if of at by for with about against between into through during
before after above below to from up down in out on off over under again further then
once here there all any both each few more most other some such no nor not only own
so than too very s t can will just don should now is am are was were be been being
have has had having do does did doing i me my we our you your he him his she her it
its they them their what which who whom whose this that these those when where why how
also as would could may might must shall
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text):
    """Case folding + accent/diacritic removal + joining things like U.S.A -> usa."""
    import unicodedata
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", text)  # u.s.a -> usa
    text = text.replace("'s", "").replace("’s", "")    # possessives: arthur's -> arthur
    return text


def tokenize(text):
    return TOKEN_RE.findall(normalize(text))


@lru_cache(maxsize=200_000)
def _stem(tok):
    return porter_stem(tok)


def analyze(text, stem=True, stop=True):
    """Return list of (term, position). Positions are token positions BEFORE
    stop-word removal, so phrase queries still line up (positional index)."""
    out = []
    for pos, tok in enumerate(tokenize(text)):
        if stop and tok in STOP_WORDS:
            continue
        out.append((_stem(tok) if stem else tok, pos))
    return out


def terms(text, stem=True, stop=True):
    return [t for t, _ in analyze(text, stem, stop)]


# ---------------------------------------------------------------------------
# Capitalised-phrase spotting, used to find candidate entity names
# ("Christopher Nolan", "Arthur's Magazine", "First for Women").
# ---------------------------------------------------------------------------
_SPAN_TOK = re.compile(r"\w[\w'’\-\.]*\w|\w|[^\w\s]")
CONNECTORS = {"of", "the", "for", "de", "van", "von", "da", "del", "la", "le", "du", "and"}
QUESTION_WORDS = {"Which", "What", "Who", "Whom", "Whose", "When", "Where", "Why", "How",
                  "Is", "Are", "Was", "Were", "Do", "Does", "Did", "In", "The", "A", "An",
                  "Have", "Has", "On", "At", "Of", "For", "From", "By", "This", "That",
                  "It", "He", "She", "They", "His", "Her", "Its", "Their", "Both"}


def capitalized_spans(text, max_len=6):
    """Maximal runs of Capitalised words (connectors like 'of'/'for' allowed inside).
    'and' is only kept inside a span if the next word is capitalised AND we are not
    splitting two names (we drop it as a connector to keep 'X and Y' separate)."""
    toks = _SPAN_TOK.findall(text)
    spans, cur = [], []

    def flush():
        while cur and cur[-1].lower() in CONNECTORS:
            cur.pop()
        while cur and cur[0] in QUESTION_WORDS:
            cur.pop(0)
        if cur:
            spans.append(" ".join(cur[:max_len]).rstrip("."))
        cur.clear()

    for i, t in enumerate(toks):
        if t[0].isupper() or (cur and t[0].isdigit()):
            cur.append(t)
        elif cur and t.lower() in CONNECTORS and t.lower() != "and" \
                and i + 1 < len(toks) and toks[i + 1][:1].isupper():
            cur.append(t)
        else:
            flush()
    flush()
    return spans
