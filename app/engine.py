# engine.py — matching with hard filters, selectable ranking, and live availability.
#
# Tutors now come from the database (via db.list_matchable_tutors), so a tutor
# who signs up and completes their profile shows up in search. Call
# rebuild_index() after any tutor profile changes to re-embed the pool.
import json
from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer, util

from . import db
from .intake import rewrite_query

MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Populated by rebuild_index(). Empty until the app calls it on startup.
tutors = []
tutor_embeddings = None
PRICE_MIN, PRICE_MAX = 0.0, 1.0

# A brand-new tutor has no reviews yet; give them a neutral rating prior so
# they aren't buried purely for being new. Adjust once real reviews exist.
NEUTRAL_RATING = 4.0

WEIGHT_PRESETS = {
    "balanced": {"semantic": 0.60, "rating": 0.25, "price": 0.15},
    "match":    {"semantic": 0.80, "rating": 0.12, "price": 0.08},
    "rating":   {"semantic": 0.50, "rating": 0.42, "price": 0.08},
    "price":    {"semantic": 0.50, "rating": 0.10, "price": 0.40},
}


def rebuild_index():
    """(Re)load matchable tutors from the DB and recompute their embeddings.

    O(n) in the number of tutors — fine at this scale. The scale path is
    incremental embedding + a vector store (pgvector), keyed off the same seam.
    """
    global tutors, tutor_embeddings, PRICE_MIN, PRICE_MAX
    tutors = db.list_matchable_tutors()
    if tutors:
        tutor_embeddings = model.encode([t["expertise"] for t in tutors])
        rates = [t["rate"] for t in tutors]
        PRICE_MIN, PRICE_MAX = min(rates), max(rates)
    else:
        tutor_embeddings = None
        PRICE_MIN, PRICE_MAX = 0.0, 1.0


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
    if not tutors:
        return {"query": query, "rewritten": query, "sort": sort,
                "total_matched": 0, "results": []}

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

    price_span = (PRICE_MAX - PRICE_MIN) or 1.0

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
        if min_rating is not None and (tutor["rating"] or 0) < min_rating:
            continue
        if only_now and not available_now:
            continue
        if use_window and not (tutor_slots & student_slots):
            continue

        # --- Re-rank survivors ---
        rating_value = tutor["rating"] if tutor["rating"] is not None else NEUTRAL_RATING
        rating_fit = rating_value / 5.0
        price_fit = clamp01((PRICE_MAX - tutor["rate"]) / price_span)
        final = (weights["semantic"] * meaning
                 + weights["rating"] * rating_fit
                 + weights["price"] * price_fit)

        results.append({
            "id": tutor["id"],
            "name": tutor["name"], "expertise": tutor["expertise"],
            "rating": tutor["rating"], "rate": tutor["rate"],
            "availability": format_availability(tutor, student_tz),
            "available_now": available_now,
            "score": round(final, 3), "meaning": round(meaning, 3),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return {"query": query, "rewritten": rewritten, "sort": sort,
            "total_matched": len(results), "results": results[:top_k]}