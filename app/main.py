import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from . import db, auth, engine

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="TutorMatch")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("TM_SECRET", "dev-insecure-secret-change-me"),
    https_only=False,
)

@app.on_event("startup")
def _startup():
    db.init()           
    engine.rebuild_index()


# Request Models
class SignupIn(BaseModel):
    email: str
    password: str
    name: str
    role: str

class LoginIn(BaseModel):
    email: str
    password: str

class ProfileIn(BaseModel):
    name: str | None = None
    role: str | None = None
    bio: str | None = None
    specialties: str | None = None
    learning: str | None = None
    education: str | None = None
    languages: str | None = None
    rate: float | None = None
    tz: int | None = None
    avail_start: int | None = None
    avail_end: int | None = None


# Helper Functions
def _current_user(request: Request):
    uid = request.session.get("uid")
    return db.get_user(uid) if uid else None

def _err(status, msg):
    return JSONResponse(status_code=status, content={"error": msg})


# Human-readable names for required fields, used in validation messages.
_FRIENDLY = {
    "specialties": "subject specialties",
    "rate": "rate",
    "tz": "timezone",
    "avail_start": "availability start",
    "avail_end": "availability end",
    "learning": "currently learning",
}

# Returns missing field for each role when creating a profile with that type
def _missing_required(role, merged):
    def blank_text(k):
        v = merged.get(k)
        return not (isinstance(v, str) and v.strip())

    def blank_num(k):
        return merged.get(k) is None

    missing = []
    if role in ("tutor", "hybrid"):
        if blank_text("specialties"):
            missing.append(_FRIENDLY["specialties"])
        for k in ("rate", "tz", "avail_start", "avail_end"):
            if blank_num(k):
                missing.append(_FRIENDLY[k])
    if role in ("student", "hybrid"):
        if blank_text("learning"):
            missing.append(_FRIENDLY["learning"])
    return missing


# Authorization API
@app.post("/api/signup")
def signup(body: SignupIn, request: Request):
    email = body.email.strip().lower()
    if "@" not in email or "." not in email:
        return _err(400, "Please enter a valid email address.")
    if len(body.password) < 8:
        return _err(400, "Password must be at least 8 characters.")
    if not body.name.strip():
        return _err(400, "Please enter your name.")
    if body.role not in db.ROLES:
        return _err(400, "Role must be student, tutor, or hybrid.")
    if db.get_user_by_email(email):
        return _err(409, "An account with that email already exists.")

    uid = db.create_user(email, auth.hash_password(body.password), body.name, body.role)
    request.session["uid"] = uid
    return db.public_view(db.get_user(uid))

@app.post("/api/login")
def login(body: LoginIn, request: Request):
    user = db.get_user_by_email(body.email)
    if not user or not auth.verify_password(body.password, user["password_hash"]):
        return _err(401, "Incorrect email or password.")
    request.session["uid"] = user["id"]
    return db.public_view(user)

@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}

@app.get("/api/me")
def me(request: Request):
    user = _current_user(request)
    if not user:
        return _err(401, "Not signed in.")
    out = db.public_view(user)
    out["email"] = user["email"]
    return out

@app.post("/api/profile")
def update_profile(body: ProfileIn, request: Request):
    user = _current_user(request)
    if not user:
        return _err(401, "Not signed in.")

    fields = body.model_dump(exclude_none=True)
    if "role" in fields and fields["role"] not in db.ROLES:
        return _err(400, "Role must be student, tutor, or hybrid.")
    for hkey in ("avail_start", "avail_end"):
        if hkey in fields and not (0 <= fields[hkey] <= 23):
            return _err(400, "Availability hours must be between 0 and 23.")
    if "tz" in fields and not (-12 <= fields["tz"] <= 14):
        return _err(400, "Timezone offset must be between -12 and +14.")
    if "rate" in fields and fields["rate"] < 0:
        return _err(400, "Rate can't be negative.")

    # Enforce role-appropriate required fields against the resulting profile,
    # so completing a profile in stages never fails on already-saved fields.
    effective_role = fields.get("role", user["role"])
    merged = {**user, **fields}
    missing = _missing_required(effective_role, merged)
    if missing:
        return _err(422, f"To save a {effective_role} profile, please fill in: {', '.join(missing)}.")

    db.update_profile(user["id"], fields)
    updated = db.get_user(user["id"])

    # If this person is (or just became) a tutor, refresh the search index.
    if updated["role"] in ("tutor", "hybrid"):
        engine.rebuild_index()

    return db.public_view(updated)


@app.get("/api/user/{uid}")
def public_profile(uid: int):
    user = db.get_user(uid)
    if not user:
        return _err(404, "No such user.")
    return db.public_view(user)


# Search API
@app.get("/search")
def search_endpoint(
    q: str, k: int = 8, sort: str = "match",
    min_price: float | None = None, max_price: float | None = None,
    min_rating: float | None = None, min_match: float | None = None,
    tz: int | None = None, only_now: bool = False,
    start: int | None = None, end: int | None = None,
):
    try:
        return engine.search(q, top_k=k, sort=sort,
                             min_price=min_price, max_price=max_price,
                             min_rating=min_rating, min_match=min_match,
                             student_tz=tz, only_now=only_now,
                             avail_start=start, avail_end=end)
    except Exception as e:
        print(f"Search failed: {e}")
        return _err(502, "The matching service is temporarily unavailable. Please try again.")


# Pages API
@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/signup")
def signup_page():
    return FileResponse(FRONTEND_DIR / "signup.html")


@app.get("/login")
def login_page():
    return FileResponse(FRONTEND_DIR / "login.html")


@app.get("/me")
def me_page():
    return FileResponse(FRONTEND_DIR / "me.html")


@app.get("/u/{uid}")
def profile_page(uid: int):
    return FileResponse(FRONTEND_DIR / "profile.html")