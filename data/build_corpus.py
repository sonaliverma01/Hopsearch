"""Turn the HotpotQA distractor file into our collection.

Input : hotpot_dev_distractor_v1.json (or train file), a list of questions; each has
        10 context paragraphs [title, [sentences]] of which 2 are gold
        (listed in supporting_facts).
Output: corpus.jsonl     one line per UNIQUE paragraph  {"doc_id", "title", "text"}
        questions.jsonl  {"qid", "question", "answer", "type", "gold_titles", "split"}

All paragraphs of the first N questions are pooled into ONE collection, so a
question must be answered by searching the whole collection, not just its own 10.
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/hotpot_dev_distractor_v1.json")
    ap.add_argument("--n", type=int, default=2000, help="number of questions to use")
    ap.add_argument("--dev", type=int, default=200, help="questions used for tuning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()

    with open(a.input, encoding="utf-8") as f:
        data = json.load(f)
    random.Random(a.seed).shuffle(data)
    data = data[:a.n]

    titles = {}
    with open(f"{a.outdir}/corpus.jsonl", "w", encoding="utf-8") as fc, \
         open(f"{a.outdir}/questions.jsonl", "w", encoding="utf-8") as fq:
        for i, ex in enumerate(data):
            for title, sents in ex["context"]:
                if title not in titles:
                    titles[title] = len(titles)
                    text = " ".join(s.strip() for s in sents)
                    fc.write(json.dumps({"doc_id": titles[title], "title": title,
                                         "text": text}, ensure_ascii=False) + "\n")
            gold = sorted({t for t, _ in ex["supporting_facts"]})
            fq.write(json.dumps({"qid": ex["_id"], "question": ex["question"],
                                 "answer": ex["answer"], "type": ex["type"],
                                 "gold_titles": gold,
                                 "split": "dev" if i < a.dev else "test"},
                                ensure_ascii=False) + "\n")
    print(f"{len(data)} questions, {len(titles)} unique paragraphs -> {a.outdir}/")


if __name__ == "__main__":
    main()
