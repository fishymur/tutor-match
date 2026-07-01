# engine.py — the matching logic, importable by both the CLI and the web app.
import json
from sentence_transformers import SentenceTransformer, util
from intake import rewrite_query

MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Load the tutor catalog from a data file — data lives separately from logic,
# so the catalog can grow (or move to a database) without touching this code.
with open("tutors.json", "r", encoding="utf-8") as f:
    tutors = json.load(f)

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