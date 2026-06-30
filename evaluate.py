from sentence_transformers import SentenceTransformer, util
from intake import rewrite_query

# Model selection
MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Tutor list with their expertise 
tutors = [
    {"name": "Ada",         "expertise": "probability theory, measure theory, stochastic processes, random variables"},
    {"name": "Bayes",       "expertise": "Bayesian statistics, statistical inference, MCMC sampling, probabilistic modeling"},
    {"name": "Markov",      "expertise": "stochastic processes, Markov chains, queueing theory, random walks"},
    {"name": "Grace",       "expertise": "real analysis, calculus, limits, continuity, sequences and series"},
    {"name": "Euler",       "expertise": "differential equations, dynamical systems, mathematical physics"},
    {"name": "Fourier",     "expertise": "Fourier analysis, partial differential equations, signal processing, harmonic analysis"},
    {"name": "Noether",     "expertise": "abstract algebra, group theory, rings and fields, Galois theory"},
    {"name": "Turing",      "expertise": "theory of computation, complexity classes, automata, computability"},
    {"name": "Dijkstra",    "expertise": "algorithms, graph theory, shortest paths, data structures"},
    {"name": "Knuth",       "expertise": "combinatorics, discrete mathematics, analysis of algorithms"},
    {"name": "Hopper",      "expertise": "compilers, programming language design, parsers, type systems"},
    {"name": "Linus",       "expertise": "operating systems, C programming, concurrency, multithreading"},
    {"name": "Lamport",     "expertise": "distributed systems, consensus algorithms, logical clocks, fault tolerance"},
    {"name": "Hinton",      "expertise": "deep learning, neural networks, backpropagation, training and optimization"},
    {"name": "Vapnik",      "expertise": "statistical learning theory, support vector machines, generalization, model complexity"},
    {"name": "Pearl",       "expertise": "causal inference, graphical models, Bayesian networks, probabilistic reasoning"},
    {"name": "Shannon",     "expertise": "information theory, entropy, coding theory, signal processing"},
    {"name": "Codd",        "expertise": "relational databases, SQL, query optimization, indexing"},
    {"name": "Nash",        "expertise": "game theory, convex optimization, equilibria, linear programming"},
    {"name": "Berners-Lee", "expertise": "web development, HTTP, REST APIs, frontend and backend"},
]

# Evaluation query set alongside correct tutor labeled
test_set = [
    {"query": "I want to prove smoking causes cancer, not just that they correlate", "correct": ["Pearl"]},
    {"query": "teaching a computer to tell cats from dogs in photos",                "correct": ["Hinton"]},
    {"query": "adding more features made my model worse, why",                       "correct": ["Vapnik"]},
    {"query": "how do I show my code stays fast as the input grows huge",            "correct": ["Knuth", "Dijkstra"]},
    {"query": "picking the best strategy when my competitor reacts to me too",       "correct": ["Nash"]},
    {"query": "a number for how unpredictable or messy some data is",               "correct": ["Shannon"]},
    {"query": "breaking a sound wave into the frequencies it's made of",            "correct": ["Fourier"]},
    {"query": "spreading one computation over many servers that might crash",        "correct": ["Lamport"]},
    {"query": "two of my threads are stuck waiting on each other forever",          "correct": ["Linus"]},
    {"query": "a function that never suddenly jumps in value",                      "correct": ["Grace"]},
    {"query": "what stays the same when you rotate or reflect a shape",             "correct": ["Noether"]},
    {"query": "my simulation gives a different answer each run, how many samples",   "correct": ["Bayes", "Markov"]},
    {"query": "is there any question a computer fundamentally can't answer",        "correct": ["Turing"]},
    {"query": "turning my source code into something the machine can run",          "correct": ["Hopper"]},
    {"query": "long-run behavior of a system that hops randomly between states",     "correct": ["Markov"]},
    {"query": "rigorous foundations of probability built on measure theory",        "correct": ["Ada"]},
    {"query": "my web page needs to save and load user accounts somewhere",         "correct": ["Berners-Lee", "Codd"]},
    {"query": "organizing records so lookups don't get slow",                       "correct": ["Codd"]},
    {"query": "finding the lowest point of a smooth cost function with limits",      "correct": ["Nash", "Hinton"]},
    {"query": "structure learning for a network of cause-and-effect variables",      "correct": ["Pearl"]},
]

# Verify labeled tutor in the test set is a tutor in the tutor list
tutor_names = {t["name"] for t in tutors}
for case in test_set:
    for name in case["correct"]:
        if name not in tutor_names:
            raise SystemExit(f"Label error: '{name}' is not a tutor name.")

# Encode tutor specialties
tutor_embeddings = model.encode([t["expertise"] for t in tutors])

recall_1 = 0
recall_3 = 0
reciprocal_ranks = []

# Evaluate the cases from the labeled tests
print(f"Evaluating model: {MODEL_NAME}\n")
for case in test_set:
    # Uses API to rewrite the query into accurate specific phrases for the embedding
    rewritten = rewrite_query(case["query"])
    q_emb = model.encode(rewritten)
    print(f"  rewrote: {case['query'][:40]}  ->  {rewritten[:60]}")
    scores = util.cos_sim(q_emb, tutor_embeddings)[0].tolist()
    ranked = sorted(zip(scores, tutors), key=lambda p: p[0], reverse=True)
    names = [t["name"] for _, t in ranked]
    ranks = []
    for n in case["correct"]:
        position = names.index(n) + 1   # +1: convert 0-based to human rank
        ranks.append(position)
    best_rank = min(ranks)

    # Evaluate recalls and reciprocal values for MRR
    if best_rank == 1:
        recall_1 += 1
    if best_rank <= 3:
        recall_3 += 1
    reciprocal_ranks.append(1.0 / best_rank)

    # Clean output with clarifying messages on failure
    mark = "OK " if best_rank <= 3 else "MISS"
    print(f"  [{mark}] rank {best_rank}: {case['query'][:50]:<50} -> {case['correct']}")
    if best_rank > 3:
        print("         who beat them:")
        for rank, (s, t) in enumerate(ranked[:5], start=1):
            star = "  <-- wanted" if t["name"] in case["correct"] else ""
            print(f"           {rank}. {s:.3f}  {t['name']}{star}")

# Show scores 
n = len(test_set)
print(f"\n--- Results for {MODEL_NAME} ---")
print(f"Recall@1: {recall_1/n:.3f}  ({recall_1}/{n})")
print(f"Recall@3: {recall_3/n:.3f}  ({recall_3}/{n})")
print(f"MRR:      {sum(reciprocal_ranks)/n:.3f}")