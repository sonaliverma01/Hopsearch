"""Evaluation measures (Lecture 8). Relevance is binary: the 2 gold paragraphs of
each HotpotQA question are relevant, everything else is not."""
import math


def precision_at_k(ranking, gold, k):
    return sum(1 for d in ranking[:k] if d in gold) / k


def recall_at_k(ranking, gold, k):
    return sum(1 for d in ranking[:k] if d in gold) / len(gold)


def both_at_k(ranking, gold, k):
    """1 if ALL gold paragraphs are in the top-k (the key multi-hop measure)."""
    return float(set(gold) <= set(ranking[:k]))


def average_precision(ranking, gold):
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, 1):
        if d in gold:
            hits += 1
            total += hits / i
    return total / len(gold)


def ndcg_at_k(ranking, gold, k):
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranking[:k], 1) if d in gold)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def interpolated_pr(ranking, gold, levels=tuple(i / 10 for i in range(11))):
    """11-point interpolated precision for one query."""
    pts, hits = [], 0
    for i, d in enumerate(ranking, 1):
        if d in gold:
            hits += 1
        pts.append((hits / len(gold), hits / i))
    return [max([p for r, p in pts if r >= lvl], default=0.0) for lvl in levels]


def evaluate(rankings, golds):
    """rankings, golds: lists aligned per question. Returns averaged metrics."""
    n = len(rankings)
    m = {}
    for k in (2, 5, 10):
        m[f"R@{k}"] = sum(recall_at_k(r, g, k) for r, g in zip(rankings, golds)) / n
        m[f"Both@{k}"] = sum(both_at_k(r, g, k) for r, g in zip(rankings, golds)) / n
    m["P@2"] = sum(precision_at_k(r, g, 2) for r, g in zip(rankings, golds)) / n
    m["P@5"] = sum(precision_at_k(r, g, 5) for r, g in zip(rankings, golds)) / n
    m["MAP"] = sum(average_precision(r, g) for r, g in zip(rankings, golds)) / n
    m["nDCG@10"] = sum(ndcg_at_k(r, g, 10) for r, g in zip(rankings, golds)) / n
    return m
