# HopSearch — a transparent multi-hop search agent

Answers questions that need **two searches in a chain** (e.g. *"Which university did the director
of Interstellar study at?"*: Interstellar → **Christopher Nolan** → UCL). It returns ranked evidence
paragraphs plus the full search trail, with no generated answer. Every IR component (inverted index,
Porter stemmer, Boolean merge, lnc.ltc cosine, zones, heap top-K, champion lists, evaluation) is
written from scratch in plain Python.

**Requirements:** Python 3.9+. No packages needed. `matplotlib` is optional and only draws the P-R curve.

## Run it (about 10 minutes)

```bash
# 1. Get the data (either way works):
#    a) pip install datasets  then  python data/from_huggingface.py
#    b) download hotpot_dev_distractor_v1.json (~45 MB) into data/ from
#       http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

# 2. Build the collection: 2,000 questions -> ~19k unique Wikipedia paragraphs
python data/build_corpus.py --input data/hotpot_dev_distractor_v1.json --n 2000

# 3. Build the inverted index (~15 s)
python -m ir.indexer

# 4. Run every experiment -> results/results.md, results.csv, pr_curve.png
python -m eval.run_experiments --max-test 500

# 5. Demo UI -> open http://localhost:8000
python app.py
```

No dataset yet? Run `python data/make_sample.py`, then step 2 with
`--input data/sample_hotpot.json --dev 3`, and steps 3–5. This uses a 10-question toy set and is only
for checking that everything runs. Use the real file for the report.

On Windows use `python`; on Mac/Linux you may need `python3`. Always run commands from the
`hopsearch/` folder.

## How it works

```
Question ─► decompose (bridge / comparison)
   bridge:     hop 1: tf-idf search for the full question
               └► bridge entities: capitalised names in the top-3 paragraphs that are
                  TITLES of other paragraphs (dictionary lookup + edit distance)
               hop 2: query = bridge entity + residual question terms
               fuse:  chain score = α·s1 + (1-α)·s2·conf  → ranked paragraphs
   comparison: one sub-query per entity, fused with Reciprocal Rank Fusion
```

## Where each lecture concept is used

| Lecture | Concept | File |
|---|---|---|
| 1 Boolean retrieval | Inverted index, postings merge, AND/OR/NOT, rarest-term-first | `ir/indexer.py`, `ir/boolean.py` |
| 2 Vocabulary | Document unit (paragraph), tokenisation, case folding, accents, stop words, Porter stemmer, positional index, phrase queries | `ir/preprocess.py`, `ir/porter.py`, `ir/boolean.py` |
| 3 Dictionaries | Title dictionary (hash), Levenshtein edit distance for fuzzy names | `agent/bridge.py` |
| 6 tf-idf / VSM | log-tf, idf, cosine, length normalisation, **lnc.ltc**, Jaccard | `ir/scorer.py` |
| 7 Scoring | Zone index (title/body), heap top-K, index elimination, champion lists, g(d) + net score, query parser | `ir/scorer.py`, `ir/indexer.py`, `agent/decompose.py` |
| 8 Evaluation | P@k, Recall@k, MAP, nDCG, 11-point interpolated P-R curve | `eval/metrics.py` |
| Beyond syllabus | Reciprocal Rank Fusion, multi-hop decomposition | `agent/pipeline.py` |

## Experiments (`eval/run_experiments.py`)

E1 baseline vs agent · E2 bridge vs comparison · E3 title weight (tuned on dev), stemming and
stop-word ablations · E4 lnc.ltc vs tf vs Jaccard · E5 chain fusion vs RRF, α · E6 full scoring vs
index elimination vs champion lists vs g(d) · E7 failure analysis · E8 "shortcut" questions.
The main metric is **Both@k**: the share of questions with *both* gold paragraphs in the top-k.

## Files

```
data/build_corpus.py   HotpotQA -> corpus.jsonl + questions.jsonl (dev/test split)
data/make_sample.py    tiny test set in HotpotQA format
ir/                    preprocess, porter, indexer, boolean, scorer
agent/                 decompose, bridge, pipeline
eval/                  metrics, run_experiments
app.py                 web demo (standard library http.server)
```
