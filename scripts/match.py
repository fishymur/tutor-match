import json

with open("tutors.json", "r", encoding="utf-8") as f:
    tutors = json.load(f)

# Only exact overlaps
def match(query, tutors):
    query_words = set(query.lower().split())
    scored = []
    for t in tutors:
        expertise_words = set(t["expertise"].lower().replace(",", "").split())
        overlap = len(query_words & expertise_words)
        scored.append((overlap, t["name"], t["expertise"]))
    scored.sort(key=lambda p: p[0], reverse=True)
    return scored


if __name__ == "__main__":
    query = "conditional expectation"
    print(f'Keyword baseline for: "{query}"\n')
    for score, name, expertise in match(query, tutors)[:5]:
        print(f"{score} shared word(s) - {name}: {expertise}")