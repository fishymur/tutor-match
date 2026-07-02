"""Benchmark harness for the tutor-matching engine.

Runs against the SAME tutor data the live app serves (tutors.json) — no more
hardcoded copies that drift out of sync.

Two arms are measured so the README's comparison is reproducible:
  1. "semantic"  — embed the raw student query directly.
  2. "intake"    — rewrite the query with the LLM intake agent first, then embed.

The intake arm only runs if an ANTHROPIC_API_KEY is available (it makes API
calls). The semantic arm needs only the embedding model, so it always runs.

Usage:
    python3 evaluate.py                       # both arms if key is set, else semantic only
    python3 evaluate.py --no-intake           # semantic arm only (no API calls)
    python3 evaluate.py --model all-MiniLM-L6-v2   # try a different embedding model
"""
import os
import json
import argparse

from sentence_transformers import SentenceTransformer, util

# --- Data: the single source of truth, identical to what the engine serves ---
with open("tutors.json", "r", encoding="utf-8") as f:
    tutors = json.load(f)

# Evaluation query set, each labeled with the tutor(s) that should surface.
# Queries are phrased the vague, plain-English way a student actually would.
test_set = [
    {"query": "I want to prove smoking causes cancer, not just that they correlate", "correct": ["Pearl"]},
    {"query": "teaching a computer to tell cats from dogs in photos",                "correct": ["Hinton"]},
    {"query": "adding more features made my model worse, why",                       "correct": ["Vapnik"]},
    {"query": "how do I show my code stays fast as the input grows huge",            "correct": ["Knuth", "Dijkstra"]},
    {"query": "picking the best strategy when my competitor reacts to me too",       "correct": ["Nash"]},
    {"query": "a number for how unpredictable or messy some data is",                "correct": ["Shannon"]},
    {"query": "breaking a sound wave into the frequencies it's made of",             "correct": ["Fourier"]},
    {"query": "spreading one computation over many servers that might crash",        "correct": ["Lamport"]},
    {"query": "two of my threads are stuck waiting on each other forever",           "correct": ["Linus"]},
    {"query": "a function that never suddenly jumps in value",                       "correct": ["Grace"]},
    {"query": "what stays the same when you rotate or reflect a shape",              "correct": ["Noether"]},
    {"query": "my simulation gives a different answer each run, how many samples",   "correct": ["Bayes", "Markov"]},
    {"query": "is there any question a computer fundamentally can't answer",         "correct": ["Turing"]},
    {"query": "turning my source code into something the machine can run",           "correct": ["Hopper"]},
    {"query": "long-run behavior of a system that hops randomly between states",     "correct": ["Markov"]},
    {"query": "rigorous foundations of probability built on measure theory",         "correct": ["Ada"]},
    {"query": "my web page needs to save and load user accounts somewhere",          "correct": ["Berners-Lee", "Codd"]},
    {"query": "organizing records so lookups don't get slow",                        "correct": ["Codd"]},
    {"query": "finding the lowest point of a smooth cost function with limits",      "correct": ["Nash", "Hinton"]},
    {"query": "structure learning for a network of cause-and-effect variables",      "correct": ["Pearl"]},
]

# Fail fast if a label no longer matches a tutor in the data.
tutor_names = {t["name"] for t in tutors}
for case in test_set:
    for name in case["correct"]:
        if name not in tutor_names:
            raise SystemExit(f"Label error: '{name}' is not a tutor in tutors.json.")


def score_query(query_text, tutor_embeddings):
    """Return (score, tutor) pairs ranked by cosine similarity to the given text."""
    q_emb = model.encode(query_text)
    scores = util.cos_sim(q_emb, tutor_embeddings)[0].tolist()
    return sorted(zip(scores, tutors), key=lambda p: p[0], reverse=True)


def run_arm(name, transform, tutor_embeddings, verbose=True):
    """Evaluate one arm. `transform` maps a raw query to the text we embed."""
    recall_1 = recall_3 = 0
    reciprocal_ranks = []
    if verbose:
        print(f"\n=== Arm: {name} ({MODEL_NAME}) ===")
    for case in test_set:
        text = transform(case["query"])
        ranked = score_query(text, tutor_embeddings)
        names = [t["name"] for _, t in ranked]
        best_rank = min(names.index(n) + 1 for n in case["correct"])  # +1: 0-based -> human rank

        if best_rank == 1:
            recall_1 += 1
        if best_rank <= 3:
            recall_3 += 1
        reciprocal_ranks.append(1.0 / best_rank)

        if verbose:
            mark = "OK  " if best_rank <= 3 else "MISS"
            print(f"  [{mark}] rank {best_rank}: {case['query'][:50]:<50} -> {case['correct']}")
            if best_rank > 3:
                print("         who beat them:")
                for r, (s, t) in enumerate(ranked[:5], start=1):
                    star = "  <-- wanted" if t["name"] in case["correct"] else ""
                    print(f"           {r}. {s:.3f}  {t['name']}{star}")

    n = len(test_set)
    return {
        "arm": name,
        "recall@1": recall_1 / n,
        "recall@3": recall_3 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "counts": (recall_1, recall_3, n),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the tutor-matching engine.")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="sentence-transformers model name")
    parser.add_argument("--no-intake", action="store_true", help="skip the LLM intake arm (no API calls)")
    args = parser.parse_args()

    MODEL_NAME = args.model
    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    tutor_embeddings = model.encode([t["expertise"] for t in tutors])
    print(f"Benchmarking against {len(tutors)} tutors, {len(test_set)} queries.")

    results = [run_arm("semantic", lambda q: q, tutor_embeddings)]

    want_intake = not args.no_intake and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if want_intake:
        from intake import rewrite_query  # imported lazily: constructing the client needs the key
        results.append(run_arm("intake", rewrite_query, tutor_embeddings))
    elif not args.no_intake:
        print("\n(Skipping intake arm: ANTHROPIC_API_KEY is not set. "
              "Set it to measure the LLM-rewrite arm.)")

    print(f"\n--- Summary ({MODEL_NAME}, {len(tutors)} tutors) ---")
    print(f"{'arm':<10} {'Recall@1':>9} {'Recall@3':>9} {'MRR':>7}")
    for r in results:
        print(f"{r['arm']:<10} {r['recall@1']:>9.3f} {r['recall@3']:>9.3f} {r['mrr']:>7.3f}")