"""Finds good questions for the live demo: ones where the single search FAILS to put
both gold paragraphs in the top 2 but the multi-hop agent SUCCEEDS.

    python find_demo_questions.py            -> prints ~15 questions, writes demo_questions.txt
"""
import json

from agent.pipeline import MultiHopAgent, baseline
from ir.indexer import InvertedIndex
from ir.scorer import Scorer

idx = InvertedIndex.load("data/index.pkl")
s = Scorer(idx, w_title=0.2, w_body=0.8)
agent = MultiHopAgent(s, alpha=0.5)
title2id = {d["title"]: i for i, d in enumerate(idx.docs)}

wins = []
for line in open("data/questions.jsonl", encoding="utf-8"):
    q = json.loads(line)
    if q["type"] != "bridge" or len(q["question"]) > 110:
        continue
    gold = {title2id[t] for t in q["gold_titles"] if t in title2id}
    if len(gold) != 2:
        continue
    base_ok = gold <= set(baseline(s, q["question"], 2))
    agent_ok = gold <= set(agent.search(q["question"], 2)["ranking"])
    if agent_ok and not base_ok:
        wins.append(q)
    if len(wins) >= 15:
        break

with open("demo_questions.txt", "w", encoding="utf-8") as f:
    for q in wins:
        line = f"{q['question']}   [answer: {q['answer']} | gold: {', '.join(q['gold_titles'])}]"
        print(line)
        f.write(line + "\n")
print(f"\n{len(wins)} questions where the agent wins -> demo_questions.txt")
