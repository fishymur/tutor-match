import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-mpnet-base-v2")

with open("tutors.json", "r", encoding="utf-8") as f:
    tutors = json.load(f)

tutor_embeddings = model.encode([t["expertise"] for t in tutors])

# Would be replaced by UI  collection 
STUDENT_TZ = -8
STUDENT_TIME = (18, 21)
STUDENT_MAX_BUDGET = 50
WEIGHTS = {"semantic": 0.70, "rating": 0.20, "price": 0.10}
TOP_K = 5

def utc_slots(start, end, tz):
    """Each available local hour converted to UTC, as a set of ints 0-23."""
    return {(hour - tz) % 24 for hour in range(start, end)}

def clamp01(x):
    return max(0.0, min(1.0, x))

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
        overlap = tutor["slots"] & student_slots
        if not overlap:
            removed += 1
            continue
        meaning_fit = clamp01(semantic)
        rating_fit = tutor["rating"] / 5.0
        price_fit = clamp01((STUDENT_MAX_BUDGET - tutor["rate"]) / STUDENT_MAX_BUDGET)
        final = (WEIGHTS["semantic"] * meaning_fit
                 + WEIGHTS["rating"] * rating_fit
                 + WEIGHTS["price"] * price_fit)
        results.append({"tutor": tutor, "final": final, "meaning": meaning_fit,
                        "rating": rating_fit, "price": price_fit, "overlap": len(overlap)})
    results.sort(key=lambda r: r["final"], reverse=True)
    print(f'\nTop {TOP_K} for "{query}"  '
          f'(budget ${STUDENT_MAX_BUDGET}, your time {STUDENT_TIME[0]}:00-{STUDENT_TIME[1]}:00):')
    print(f"  ({removed} of {len(tutors)} tutors removed — unavailable in your window)")
    for r in results[:TOP_K]:
        t = r["tutor"]
        print(f"  {r['final']:.3f}  {t['name']:<12} "
              f"(meaning {r['meaning']:.2f} · rating {r['rating']:.2f} · price {r['price']:.2f})  "
              f"${t['rate']}/hr {t['rating']}star · {r['overlap']}h overlap")