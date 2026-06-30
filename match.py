tutors = [
    {"name": "Ada",   "expertise": "probability theory, measure theory, stochastic processes"},
    {"name": "Linus", "expertise": "operating systems, C programming, concurrency"},
    {"name": "Grace", "expertise": "linear algebra, calculus, real analysis"},
]

def match(query, tutors):
    query_words = set(query.lower().split())
    scored = []
    for t in tutors:
        expertise_words = set(t["expertise"].lower().replace(",", "").split())
        overlap = len(query_words & expertise_words)
        scored.append((overlap, t["name"], t["expertise"]))
    scored.sort(reverse=True)
    return scored

query = "conditional expectation"
for score, name, expertise in match(query, tutors):
    print(f"{score} points — {name}: {expertise}")