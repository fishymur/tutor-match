# app.py — wraps the matching engine in a web API and serves the frontend.
from fastapi import FastAPI
from fastapi.responses import FileResponse
from engine import search

app = FastAPI(title="TutorMatch")

@app.get("/search")
def search_endpoint(q: str, k: int = 5):
    """GET /search?q=your+question  ->  ranked tutors as JSON."""
    return search(q, top_k=k)

@app.get("/")
def home():
    """Serve the search page at the root URL."""
    return FileResponse("index.html")