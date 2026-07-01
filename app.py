# app.py — wraps the matching engine in a web API and serves the frontend.
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from engine import search

app = FastAPI(title="TutorMatch")

@app.get("/search")
def search_endpoint(q: str, k: int = 5):
    """GET /search?q=your+question  ->  ranked tutors as JSON."""
    try:
        return search(q, top_k=k)
    except Exception as e:
        # Log the real error for us; show the user something friendly, not internals.
        print(f"Search failed: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": "The matching service is temporarily unavailable. Please try again."},
        )

@app.get("/")
def home():
    """Serve the search page at the root URL."""
    return FileResponse("index.html")