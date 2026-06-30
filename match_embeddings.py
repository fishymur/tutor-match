from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-mpnet-base-v2")

# Each tutor now has: tz (their UTC offset) and hours (start, end) in THEIR local time.
tutors = [
    {"name": "Ada",      "expertise": "probability theory, measure theory, stochastic processes, random variables", "rating": 4.9, "rate": 35, "tz": -8, "hours": (17, 22)},
    {"name": "Bayes",    "expertise": "Bayesian statistics, statistical inference, MCMC sampling, probabilistic modeling", "rating": 4.4, "rate": 60, "tz": 1,  "hours": (9, 18)},
    {"name": "Grace",    "expertise": "real analysis, calculus, limits, continuity, sequences and series", "rating": 4.7, "rate": 30, "tz": -8, "hours": (8, 12)},
    {"name": "Noether",  "expertise": "abstract algebra, group theory, rings and fields, Galois theory", "rating": 4.8, "rate": 45, "tz": 9,  "hours": (11, 15)},
    {"name": "Euler",    "expertise": "differential equations, dynamical systems, mathematical physics", "rating": 4.5, "rate": 40, "tz": -8, "hours": (13, 18)},
    {"name": "Turing",   "expertise": "theory of computation, complexity, automata, computability", "rating": 4.6, "rate": 50, "tz": -5, "hours": (20, 24)},
    {"name": "Dijkstra", "expertise": "algorithms, graph theory, shortest paths, data structures", "rating": 4.9, "rate": 55, "tz": -5, "hours": (19, 23)},
    {"name": "Knuth",    "expertise": "combinatorics, discrete mathematics, analysis of algorithms", "rating": 5.0, "rate": 70, "tz": 1,  "hours": (9, 17)},
    {"name": "Hopper",   "expertise": "compilers, programming language design, parsers, type systems", "rating": 4.3, "rate": 48, "tz": -8, "hours": (18, 22)},
    {"name": "Linus",    "expertise": "operating systems, C programming, concurrency, multithreading", "rating": 4.2, "rate": 38, "tz": 1,  "hours": (20, 24)},
    {"name": "Hinton",   "expertise": "deep learning, neural networks, backpropagation, training and optimization", "rating": 4.7, "rate": 65, "tz": 9,  "hours": (10, 16)},
    {"name": "Shannon",  "expertise": "information theory, entropy, coding theory, signal processing", "rating": 4.6, "rate": 52, "tz": -5, "hours": (18, 22)},
]

tutor_embeddings = model.encode([t["expertise"] for t in tutors])

# ---- The student's situation (your design knobs) ----
STUDENT_TZ = -8             # student in UTC-8 (California)
STUDENT_TIME = (18, 21)    # wants a session 6pm-9pm THEIR local time
STUDENT_MAX_BUDGET = 50
WEIGHTS = {"semantic": 0.60, "rating": 0.25, "price": 0.15}
TOP_K = 5

def utc_slots(start, end, tz):
    """Each available hour, converted to UTC, as a set of integers 0-23."""
    return {(hour - tz) % 24 for hour in range(start, end)}

def clamp01(x):
    return max(0.0, min(1.0, x))

# Convert the student's window, and every tutor's window, to UTC once up front.
student_slots = utc_slots(STUDENT_TIME[0], STUDENT_TIME[1], STUDENT_TZ)
for t in tutors:
    t["slots"] = utc_slots(t["hours"][0], t["hours"][1], t["tz"])

while True:
    query = input("\nWhat do you need help with? (type 'quit' to stop)\n> ")
    if query.strip().lower() in {"quit", "exit", ""}:
        break

    query_embedding = model.encode(query)
    semantic_scores = util.cos_sim(query_embedding, tutor_embeddings)[0].tolist()

    results = []
    removed = 0
    for tutor, semantic in zip(tutors, semantic_scores):
        # HARD FILTER: no overlapping hours -> drop this tutor, don't even score them.
        overlap = tutor["slots"] & student_slots
        if not overlap:
            removed += 1
            continue

        meaning_fit = clamp01(semantic)
        rating_fit  = tutor["rating"] / 5.0
        price_fit   = clamp01((STUDENT_MAX_BUDGET - tutor["rate"]) / STUDENT_MAX_BUDGET)
        final = (WEIGHTS["semantic"] * meaning_fit
               + WEIGHTS["rating"]   * rating_fit
               + WEIGHTS["price"]    * price_fit)

        results.append({"tutor": tutor, "final": final, "meaning": meaning_fit,
                        "rating": rating_fit, "price": price_fit, "overlap": len(overlap)})

    results.sort(key=lambda r: r["final"], reverse=True)

    print(f'\nTop {TOP_K} for "{query}"  '
          f'(budget ${STUDENT_MAX_BUDGET}, your time {STUDENT_TIME[0]}:00-{STUDENT_TIME[1]}:00):')
    print(f"  ({removed} tutors removed — unavailable in your window)")
    for r in results[:TOP_K]:
        t = r["tutor"]
        print(f"  {r['final']:.3f}  {t['name']:<9} "
              f"(meaning {r['meaning']:.2f} · rating {r['rating']:.2f} · price {r['price']:.2f})  "
              f"${t['rate']}/hr {t['rating']}★ · {r['overlap']}h overlap")