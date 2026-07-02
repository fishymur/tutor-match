"""Keyword-matching baseline.

The naive approach the rest of the project improves on: score tutors purely by
how many words their expertise literally shares with the query. It reads from
tutors.json so there is one source of truth for tutor data across the repo.

Run it to see the baseline fail on paraphrased queries (e.g. "conditional
expectation" shares no words with "probability theory, measure theory").
"""
import json

with open("tutors.json", "r", encoding="utf-8") as f:
    tutors = json.load(f)


def match(query, tutors):
    """Rank tutors by literal word overlap between query and expertise."""
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