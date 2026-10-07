"""Download HotpotQA (distractor) from Hugging Face and save it in the ORIGINAL
HotpotQA JSON format that build_corpus.py expects.

    pip install datasets
    python data/from_huggingface.py          # -> data/hotpot_dev_distractor_v1.json

Hugging Face stores context as {"title": [...], "sentences": [[...]]} and
supporting_facts as {"title": [...], "sent_id": [...]}; the original file uses
lists of [title, sentences] and [title, sent_id]. We convert back.
"""
import json

from datasets import load_dataset


def convert(ex):
    return {
        "_id": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex["type"],
        "level": ex["level"],
        "supporting_facts": [[t, s] for t, s in zip(ex["supporting_facts"]["title"],
                                                    ex["supporting_facts"]["sent_id"])],
        "context": [[t, sents] for t, sents in zip(ex["context"]["title"],
                                                   ex["context"]["sentences"])],
    }


def main():
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    data = [convert(ex) for ex in ds]
    with open("data/hotpot_dev_distractor_v1.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"saved {len(data)} questions -> data/hotpot_dev_distractor_v1.json")


if __name__ == "__main__":
    main()
