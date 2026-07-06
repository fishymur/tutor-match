---
title: TutorMatch
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Tutor Matching Algorithm

**The blueprint for a matching algorithm that matches students with specific subject needs to tutors who specialize in those requests.**

- **Live demo:** https://huggingface.co/spaces/fishymur/tutor-match
- **Source:** https://github.com/fishymur/tutor-match

With the increasing use of AI in learning, higher education has become seemingly more accessible. This, however, comes at a cost, a lack of structural learning that ignores proper prerequisite structuring and a reliance on biases that AI feeds to students who would have learned content in a much different environment otherwise. Generic tutoring platforms and educational platforms such as Khan Academy and Coursera lack involved tutoring and real conversations between specialists and learners. The aim of this algorithm is to create a platform around tutor matching for students where not just categories and tags are matched for broad needs, but more specific queries such as "why won't my neural net converge" can be processed and point to the best specialist to answer the question. The issue we aim to target is the jargon gap: students tend to describe their problems imprecisely and specifically, while tutors advertise expertise using broad, formal, and technical terminology. A keyword search fails to connect queries like “how to make my code run more efficiently” with “algorithmic complexity analysis,” even though they refer to the same concept. To bridge the semantic gap, this repository outlines a system that connects informal queries with formal tutor labels.

The surrounding design around reviews, bookings, payments, etc are deferred by design. The matching is the intent of the repo, later to be scaled

## Framework

Student Request -> [Intake Agent (LLM rewrite claude-haiku-4-5) -> Embed -> Vector Search -> Hard Categorical Filtering -> Re-rank w/ Weighted Variables] -> Rankings with Categorical Explanation

1. **Intake Agent** - Claude Haiku 4.5 rewrites vague student query and condenses into precise topics and vocabulary used by tutors before embedding.
2. **Semantic Retrieval** - All tutors in database and query are embedded into shared space; retrieval by cosine similarity.
3. **Filter** - Filter out by availability of tutors and other filters put on by users.
4. **Re-rank** - Weighted balance of semantic relevance of query to tutor, ratings, price, etc. All categories normalized and weighted accordingly.
5. **Explanation** - Outputs match's component score breakdown.

## Features
1. Hard filtering and selectable soft ranking system: Users are able to filter results by price, time, rating, and keyword similarity during their search. They are also able to change the criteria of the search by the weighting of performance, price, and similarity to the query.
2. Timezone awareness: the frontend detects the student's UTC offset and both filters and displays tutor hours in the student's local time, alongside a :availible now badge for currently active tutors.
3. Account and profile system: Users are able to select account types as students, tutors , and hybrids, where users can setup a profile to show what they are currently learning, what they are capable of teaching, and availabilities. The accounts are locked away by secure encryption using PBKDF2. 
4. Evaluation mechanism: a reproducible check for recall@1, recall@2, and MRR over a labeled dataset.

## Tech Stack
Backend: Python, FASTAPI, Uvicorn
Semantic Search: SentenceTransformer, all-mpnet-base-v2
Intake Agent: claude-haiku-4-5
Storage: SQLite
Authorization: stdlib PBKDF2
Frontend: HTML/CSS/JS
Deployment: Docker - Hugging Face

## Project Structure

```
tutor-match/
├── Dockerfile              # Container: launches the FastAPI app on port 7860
├── README.md
├── requirements.txt
├── .gitignore
├── app/                    # The web application (Python package)
│   ├── __init__.py
│   ├── main.py             # FastAPI app: routes, sessions, auth + profile API
│   ├── engine.py           # Matching: embed -> filter -> re-rank
│   ├── intake.py           # LLM query-rewrite w/ caching
│   ├── db.py               # SQLite layer: schema, CRUD, seeding, matchable-tutor query
│   └── auth.py             # Password hashing / verification
├── frontend/               # Static pages served by FastAPI
│   ├── index.html          # Search UI
│   ├── signup.html         # Account creation with role selection
│   ├── login.html
│   ├── me.html             # Profile editor
│   └── profile.html        # Public read-only profile
├── data/
│   └── tutors.json         # Seed dataset (115 tutors) labeled for evaulation harness
└── scripts/                # Offline tools (not part of the running server)
    ├── evaluate.py         # Benchmark harness (Recall@k, MRR)
    ├── match.py            # Keyword-matching baseline (shows what we improve on)
    └── match_embeddings.py # Interactive CLI matcher
```

## Running Locally

Requires Python 3.10+
```bash
pip install -r requirements.txt
 
# Session signing key + anthropic API key
export TM_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export ANTHROPIC_API_KEY="sk-ant-..."
 
uvicorn app.main:app --reload --port 7860
```
Open http://127.0.0.1:7860; downloads the embedding model and seeds from `data/tutors.json`.
 
Scripts 
```bash
python -m scripts.evaluate              # benchmark (semantic + intake arms)
python -m scripts.evaluate --no-intake  # semantic only, no API calls
python -m scripts.match                 # keyword baseline
python -m scripts.match_embeddings      # interactive matcher
```

## Failure

We first observe the most simple case in which a query fails on generic educational platforms:

**Pure Keyword Baseline** - A query like "I need help with conditional expectation and Bayes theorem" scores 0 against a tutor who specializes in "probability theory, measure theory" since there are no directly shared words between the two.

**Semantic Matching Improvement** - The previous query now scores significantly higher since we are now using a embedding space where the query sits near the expertise in terms of meaning and context.

**Adding the Intake Agent** - A informal query that would have been matched poorly by semantics alone are rewritten initially by an agent, then embedded for improved accuracy.

## Evaluation 

The match qualities recall@1, recall@3, MRR are measured on a labeled set of 20 plain english queries against a 115 tutor dataset generated by OPUS 4.8. The tutors were created intentionally with overlapping specialists and difficult differentiation to not trivially saturate the results.

**Recall@1** — Frequency of correct best query ranked first.
**Recall@3** — Frequency of correct best query ranked in the top three
**MRR** (Mean Reciprocal Rank) — Weighted in between of recall@1 and recall@3 (higher rank is weighted but contributes to score if in top 3)

| Arm | Recall@1 | Recall@3 | MRR |
|-----------------------------|:--------:|:--------:|:-----:|
| Semantic only               | 0.50     | 0.80     | 0.678 |
| Semantic + intake agent     | **0.75** | **0.90** | **0.835** |

The LLM intake increases the recall@1 score from 0.5 to 0.75, increasing the accuracy of queries that do not share surface terms with the correctly labled tutors. For example, the query: "how do I show my code stays fast as input grows" jumps from rank 10 to rank 3 for a tutor that specialized in complexity-analysis.

## Design Decisions

1. Normalization of instances before weighting for search ranking: values such as price, rating, and semantic similarity were all normalized on a 0-1 scale before getting weighted by the prefered search style so that heteogenous units do not distort the score.
2. New tutor signups are scored with a neutral score rather than a score of 0 so that they do not get burried.
3. Weighting for presets are hand-picked for general stability in search results.
4. SQLite used instead of Postgres for the time-being
5. Tutor roles only appear on search-results once required fields are filled out.


## Limitations and Roadmap

1. Match % displayed on search result is raw cosine similarity and should be scaled for the display
2. Re-embedding on tutor profile change is o(n).
3. SQLite persists locally, but problems arise for hosting. The runtime database does not survive on a rebuild so a persistent storage option or Postgres are necessary.
4. Measures retrieval quality and no the whole final ranking.