from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from engine import search

app = FastAPI(title="TutorMatch")

@app.get("/search")
def search_endpoint(
    q: str, k: int = 8, sort: str = "match",
    min_price: float | None = None, max_price: float | None = None,
    min_rating: float | None = None, min_match: float | None = None,
    tz: int | None = None, only_now: bool = False,
    start: int | None = None, end: int | None = None,
):
    try:
        return search(q, top_k=k, sort=sort,
                      min_price=min_price, max_price=max_price,
                      min_rating=min_rating, min_match=min_match,
                      student_tz=tz, only_now=only_now,
                      avail_start=start, avail_end=end)
    except Exception as e:
        print(f"Search failed: {e}")
        return JSONResponse(status_code=502,
            content={"error": "The matching service is temporarily unavailable. Please try again."})

@app.get("/")
def home():
    return FileResponse("index.html")