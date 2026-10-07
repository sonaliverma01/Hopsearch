"""Porter stemmer (Lecture 2: stemming), written from the original 1980 algorithm.

We implement it ourselves so the project has no dependencies; it gives the same
idea as nltk's PorterStemmer: strip suffixes in 5 steps so that
"studied", "studies", "studying" all become "studi".

Key idea: m = number of vowel-consonant (VC) sequences in the stem.
Rules only fire when the remaining stem is "long enough" (m > 0, m > 1, ...).
"""


def _cons(w, i):
    """True if w[i] is a consonant. 'y' is a consonant only after a vowel."""
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _m(stem):
    """Measure m: number of VC sequences in [C](VC)^m[V]."""
    n, i, L = 0, 0, len(stem)
    while i < L and _cons(stem, i):
        i += 1
    while i < L:
        while i < L and not _cons(stem, i):
            i += 1
        if i >= L:
            break
        while i < L and _cons(stem, i):
            i += 1
        n += 1
    return n


def _has_vowel(stem):
    return any(not _cons(stem, i) for i in range(len(stem)))


def _double_cons(w):
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w):
    """consonant-vowel-consonant ending, last letter not w, x or y (e.g. -hop)."""
    return (len(w) >= 3 and _cons(w, len(w) - 3) and not _cons(w, len(w) - 2)
            and _cons(w, len(w) - 1) and w[-1] not in "wxy")


STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
         ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
         ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
         ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
         ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log")]
STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
         ("ical", "ic"), ("ful", ""), ("ness", "")]
STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
         "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]
STEP2.sort(key=lambda p: -len(p[0]))
STEP3.sort(key=lambda p: -len(p[0]))
STEP4.sort(key=len, reverse=True)


def stem(w):
    if len(w) <= 2:
        return w
    # Step 1a: plurals
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # Step 1b: -eed, -ed, -ing
    fix = False
    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w, fix = w[:-2], True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w, fix = w[:-3], True
    if fix:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _double_cons(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _m(w) == 1 and _cvc(w):
            w += "e"
    # Step 1c: y -> i
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"
    # Step 2 and 3: map double suffixes to single ones (m > 0)
    for table in (STEP2, STEP3):
        for suf, rep in table:
            if w.endswith(suf):
                if _m(w[:-len(suf)]) > 0:
                    w = w[:-len(suf)] + rep
                break
    # Step 4: remove suffixes when m > 1
    for suf in STEP4:
        if w.endswith(suf):
            base = w[:-len(suf)]
            if _m(base) > 1 and (suf != "ion" or base.endswith(("s", "t"))):
                w = base
            break
    # Step 5: tidy up final -e and -ll
    if w.endswith("e"):
        base = w[:-1]
        if _m(base) > 1 or (_m(base) == 1 and not _cvc(base)):
            w = base
    if _m(w) > 1 and _double_cons(w) and w.endswith("l"):
        w = w[:-1]
    return w


if __name__ == "__main__":
    for x in ["caresses", "ponies", "studied", "studies", "studying", "university",
              "universities", "relational", "hopping", "directed", "director",
              "generalization", "nationality", "born", "films"]:
        print(f"{x:16s} -> {stem(x)}")
