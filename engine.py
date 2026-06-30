# engine.py — the matching logic, importable by both the CLI and the web app.
from sentence_transformers import SentenceTransformer, util
from intake import rewrite_query

MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Tutor catalog: name, expertise, rating (out of 5), hourly rate (USD).
tutors = [
    {"name": "Ada",         "expertise": "probability theory, measure theory, stochastic processes, random variables", "rating": 4.9, "rate": 35},
    {"name": "Bayes",       "expertise": "Bayesian statistics, statistical inference, MCMC sampling, probabilistic modeling", "rating": 4.4, "rate": 60},
    {"name": "Markov",      "expertise": "stochastic processes, Markov chains, queueing theory, random walks", "rating": 4.6, "rate": 42},
    {"name": "Grace",       "expertise": "real analysis, calculus, limits, continuity, sequences and series", "rating": 4.7, "rate": 30},
    {"name": "Euler",       "expertise": "differential equations, dynamical systems, mathematical physics", "rating": 4.5, "rate": 40},
    {"name": "Fourier",     "expertise": "Fourier analysis, partial differential equations, signal processing, harmonic analysis", "rating": 4.6, "rate": 46},
    {"name": "Noether",     "expertise": "abstract algebra, group theory, rings and fields, Galois theory", "rating": 4.8, "rate": 45},
    {"name": "Turing",      "expertise": "theory of computation, complexity classes, automata, computability", "rating": 4.6, "rate": 50},
    {"name": "Dijkstra",    "expertise": "algorithms, graph theory, shortest paths, data structures", "rating": 4.9, "rate": 55},
    {"name": "Knuth",       "expertise": "combinatorics, discrete mathematics, analysis of algorithms", "rating": 5.0, "rate": 70},
    {"name": "Hopper",      "expertise": "compilers, programming language design, parsers, type systems", "rating": 4.3, "rate": 48},
    {"name": "Linus",       "expertise": "operating systems, C programming, concurrency, multithreading", "rating": 4.2, "rate": 38},
    {"name": "Lamport",     "expertise": "distributed systems, consensus algorithms, logical clocks, fault tolerance", "rating": 4.7, "rate": 58},
    {"name": "Hinton",      "expertise": "deep learning, neural networks, backpropagation, training and optimization", "rating": 4.7, "rate": 65},
    {"name": "Vapnik",      "expertise": "statistical learning theory, support vector machines, generalization, model complexity", "rating": 4.5, "rate": 60},
    {"name": "Pearl",       "expertise": "causal inference, graphical models, Bayesian networks, probabilistic reasoning", "rating": 4.8, "rate": 62},
    {"name": "Shannon",     "expertise": "information theory, entropy, coding theory, signal processing", "rating": 4.6, "rate": 52},
    {"name": "Codd",        "expertise": "relational databases, SQL, query optimization, indexing", "rating": 4.4, "rate": 44},
    {"name": "Nash",        "expertise": "game theory, convex optimization, equilibria, linear programming", "rating": 4.7, "rate": 50},
    {"name": "Berners-Lee", "expertise": "web development, HTTP, REST APIs, frontend and backend", "rating": 4.3, "rate": 40},
]

# Embed every tutor's expertise once, when this module is first imported.
tutor_embeddings = model.encode([t["expertise"] for t in tutors])

WEIGHTS = {"semantic": 0.60, "rating": 0.25, "price": 0.15}
DEFAULT_BUDGET = 50

def clamp01(x):
    return max(0.0, min(1.0, x))

def search(query, top_k=5, budget=DEFAULT_BUDGET):
    """Rewrite, embed, score, and rank tutors for a free-text request."""
    rewritten = rewrite_query(query)
    q_emb = model.encode(rewritten)
    semantic_scores = util.cos_sim(q_emb, tutor_embeddings)[0].tolist()

    results = []
    for tutor, semantic in zip(tutors, semantic_scores):
        meaning = clamp01(semantic)
        rating_fit = tutor["rating"] / 5.0
        price_fit = clamp01((budget - tutor["rate"]) / budget)
        final = (WEIGHTS["semantic"] * meaning
               + WEIGHTS["rating"] * rating_fit
               + WEIGHTS["price"] * price_fit)
        results.append({
            "name": tutor["name"],
            "expertise": tutor["expertise"],
            "rating": tutor["rating"],
            "rate": tutor["rate"],
            "score": round(final, 3),
            "meaning": round(meaning, 3),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"query": query, "rewritten": rewritten, "results": results[:top_k]}