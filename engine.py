# engine.py — matching with hard filters, selectable ranking, and live availability.
import json
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer, util
from intake import rewrite_query

MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

with open("tutors.json", "r", encoding="utf-8") as f:
    tutors = json.load(f)

tutor_embeddings = model.encode([t["expertise"] for t in tutors])

PRICE_MIN = min(t["rate"] for t in tutors)
PRICE_MAX = max(t["rate"] for t in tutors)

WEIGHT_PRESETS = {
    "balanced": {"semantic": 0.60, "rating": 0.25, "price": 0.15},
    "match":    {"semantic": 0.80, "rating": 0.12, "price": 0.08},
    "rating":   {"semantic": 0.50, "rating": 0.42, "price": 0.08},
    "price":    {"semantic": 0.50, "rating": 0.10, "price": 0.40},
}

def clamp01(x):
    return max(0.0, min(1.0, x))

def utc_slots(start, end, tz):
    """UTC hours covered by a local [start, end) window (handles midnight wrap)."""
    hours = range(start, end) if end > start else list(range(start, 24)) + list(range(0, end))
    return {(h - tz) % 24 for h in hours}

def _h12(hour):
    """Turn a 0-23 hour into 12-hour text, e.g. 0 -> '12 AM', 13 -> '1 PM'."""
    suffix = "AM" if hour < 12 else "PM"
    h = hour % 12
    if h == 0:
        h = 12
    return f"{h} {suffix}"

def format_availability(tutor, student_tz):
    """Human-readable availability, shown in the student's timezone if known."""
    h0, h1 = tutor["hours"]
    ttz = tutor["tz"]
    if student_tz is not None:
        shift = student_tz - ttz
        s, e = (h0 + shift) % 24, (h1 + shift) % 24
        return f"{_h12(s)} - {_h12(e)} your time"
    label = f"UTC{'+' if ttz >= 0 else '-'}{abs(ttz)}"
    return f"{_h12(h0)} - {_h12(h1)} {label}"

def search(query, top_k=8, sort="match", min_price=None, max_price=None,
           min_rating=None, min_match=None, student_tz=None,
           only_now=False, avail_start=None, avail_end=None):
    """Rewrite, embed, filter, score, and rank tutors for a request."""
    rewritten = rewrite_query(query)
    q_emb = model.encode(rewritten)
    semantic_scores = util.cos_sim(q_emb, tutor_embeddings)[0].tolist()

    weights = WEIGHT_PRESETS.get(sort, WEIGHT_PRESETS["match"])

    # Current UTC hour — drives both the "available now" filter and the badge.
    now_utc_hour = datetime.now(timezone.utc).hour

    use_window = (student_tz is not None
                  and avail_start is not None and avail_end is not None)
    if use_window:
        student_slots = utc_slots(avail_start, avail_end, student_tz)

    results = []
    for tutor, semantic in zip(tutors, semantic_scores):
        meaning = clamp01(semantic)
        tutor_slots = utc_slots(tutor["hours"][0], tutor["hours"][1], tutor["tz"])
        available_now = now_utc_hour in tutor_slots

        # --- HARD FILTERS ---
        if min_match is not None and meaning * 100 < min_match:
            continue
        if min_price is not None and tutor["rate"] < min_price:
            continue
        if max_price is not None and tutor["rate"] > max_price:
            continue
        if min_rating is not None and tutor["rating"] < min_rating:
            continue
        if only_now and not available_now:
            continue
        if use_window and not (tutor_slots & student_slots):
            continue

        # --- Re-rank survivors ---
        rating_fit = tutor["rating"] / 5.0
        price_fit = clamp01((PRICE_MAX - tutor["rate"]) / (PRICE_MAX - PRICE_MIN))
        final = (weights["semantic"] * meaning
               + weights["rating"] * rating_fit
               + weights["price"] * price_fit)

        results.append({
            "name": tutor["name"], "expertise": tutor["expertise"],
            "rating": tutor["rating"], "rate": tutor["rate"],
            "availability": format_availability(tutor, student_tz),
            "available_now": available_now,
            "score": round(final, 3), "meaning": round(meaning, 3),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"query": query, "rewritten": rewritten, "sort": sort,
            "total_matched": len(results), "results": results[:top_k]}