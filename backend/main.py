import json, os, logging
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base
from ml.pipeline import PROC, REPORTS, ROOT, process
from ml.recs import Recommender

log = logging.getLogger("cinematch"); logging.basicConfig(level=logging.INFO)
Base = declarative_base(); S = {}
class Item(Base):
    __tablename__ = "items"
    user_id = Column(String, primary_key=True); movie_id = Column(Integer, primary_key=True)
    status = Column(String, nullable=False, default="list"); rating = Column(Float, nullable=True)
class Feedback(BaseModel):
    status: str | None = Field(None, pattern="^(list|watched|disliked)$")
    rating: float | None = Field(None, ge=1, le=5)

@asynccontextmanager
async def lifespan(app):
    if not (PROC / "movies.csv").exists(): process()
    movies, ratings = pd.read_csv(PROC / "movies.csv"), pd.read_csv(PROC / "ratings.csv")
    S.update(ratings=ratings, rec=Recommender(movies, ratings),
             quality=json.loads((REPORTS / "data_quality.json").read_text()),
             engine=create_engine(f"sqlite:///{os.environ.get('CINEMATCH_DB', ROOT / 'data/cinematch.db')}"))
    Base.metadata.create_all(S["engine"]); yield

app = FastAPI(title="CineMatch API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CINEMATCH_CORS", "http://localhost:5173").split(","), allow_methods=["*"], allow_headers=["*"])

def rows(df): return json.loads(df.astype(object).where(df.notna(), None).to_json(orient="records"))
def one(mid):
    df = S["rec"].stats; r = df[df.movie_id == mid]
    if r.empty: raise HTTPException(404, f"Movie {mid} not found")
    return rows(r)[0]
def enrich(recs): return [{**one(r["movie_id"]), "score": r["score"], "reason": r["reason"]} for r in recs]

@app.get("/api/health")
def health(): return {"status": "ok", "data_source": S["quality"]["source"], "is_sample": S["quality"]["is_sample"]}

@app.get("/api/genres")
def genres(): return sorted({g for x in S["rec"].movies.genres.dropna() for g in x.split("|") if g})

@app.get("/api/movies")
def movies(q: str | None = None, genres: str | None = None, min_year: int | None = None, max_year: int | None = None,
           min_rating: float = Query(0, ge=0, le=5), min_votes: int = Query(0, ge=0),
           sort: str = Query("popularity", pattern="^(relevance|rating|popularity|year|title)$"),
           page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=50)):
    df = S["rec"].stats
    if q: df = df[df.title.str.contains(q, case=False, regex=False) | df.genres.fillna("").str.contains(q, case=False, regex=False)]
    for g in filter(None, (genres or "").split(",")): df = df[df.genres.fillna("").str.split("|").apply(lambda L: g in L)]
    if min_year: df = df[df.year >= min_year]
    if max_year: df = df[df.year <= max_year]
    df = df[(df.avg.fillna(0) >= min_rating) & (df.votes >= min_votes)]
    key, asc = {"relevance": ("wr", False), "popularity": ("votes", False), "rating": ("wr", False),
                "year": ("year", False), "title": ("title", True)}[sort]
    df = df.sort_values(key, ascending=asc, kind="stable")
    return {"total": len(df), "page": page, "results": rows(df.iloc[(page - 1) * page_size: page * page_size])}

@app.get("/api/movies/{movie_id}")
def movie(movie_id: int): return one(movie_id)

@app.get("/api/recommendations/popular")
def popular(k: int = Query(10, ge=1, le=50)):
    return enrich([{"movie_id": int(r.movie_id), "score": float(r.wr), "reason": "Highly rated with a solid number of votes (weighted rating)"}
                   for r in S["rec"].stats.sort_values("wr", ascending=False).head(k).itertuples()])

@app.get("/api/recommendations/similar/{movie_id}")
def similar(movie_id: int, k: int = Query(10, ge=1, le=50)):
    one(movie_id); return enrich(S["rec"].similar(movie_id, k))

@app.get("/api/recommendations/user/{user_id}")
def user_recs(user_id: str, k: int = Query(10, ge=1, le=50), genres: str | None = None):
    with Session(S["engine"]) as s: items = s.query(Item).filter_by(user_id=user_id).all()
    hist = {i.movie_id: i.rating for i in items if i.rating}
    if not hist and user_id.isdigit():  # MovieLens user id
        d = S["ratings"][S["ratings"].user_id == int(user_id)]; hist = dict(zip(d.movie_id, d.rating))
    excl = [i.movie_id for i in items if i.status in ("watched", "disliked")]
    cold = not hist
    return {"cold_start": cold, "results": enrich(S["rec"].recommend(hist, [g for g in (genres or "").split(",") if g] or None, k, excl))}

@app.put("/api/users/{user_id}/movies/{movie_id}")
def set_item(user_id: str, movie_id: int, body: Feedback):
    one(movie_id)
    with Session(S["engine"]) as s:
        it = s.get(Item, (user_id, movie_id)) or Item(user_id=user_id, movie_id=movie_id, status="list")
        if body.status: it.status = body.status
        if body.rating is not None: it.rating = body.rating
        s.merge(it); s.commit()
    return {"user_id": user_id, "movie_id": movie_id, "status": it.status, "rating": it.rating,
            "note": "Used by the next /recommendations/user call; no retraining needed (content/CF computed at request time)."}

@app.get("/api/users/{user_id}/movies")
def list_items(user_id: str, status: str | None = None):
    with Session(S["engine"]) as s:
        q = s.query(Item).filter_by(user_id=user_id)
        return [{"movie_id": i.movie_id, "status": i.status, "rating": i.rating} for i in (q.filter_by(status=status) if status else q)]

@app.get("/api/providers/{movie_id}")
def providers(movie_id: int):
    m = one(movie_id)
    return {"movie_id": movie_id, "providers": [], "status": "Availability unknown",
            "note": "No verified provider data configured (TMDB key not set). Nothing is labelled free.",
            "search_links": [{"name": "JustWatch India", "url": f"https://www.justwatch.com/in/search?q={m['title']}"}]}

@app.get("/api/analytics/overview")
def overview():
    g = {}
    for x in S["rec"].movies.genres.dropna():
        for k in x.split("|"):
            if k: g[k] = g.get(k, 0) + 1
    return {**S["quality"], "genre_counts": g}

@app.get("/api/analytics/model-evaluation")
def model_eval():
    p = REPORTS / "evaluation.json"
    return json.loads(p.read_text()) if p.exists() else {"status": "not_run", "hint": "python -m ml.evaluate"}

@app.get("/", include_in_schema=False)
def index(): return FileResponse(ROOT / "frontend/index.html")


# TMDB-backed catalogue: configure TMDB_READ_ACCESS_TOKEN in the deployment environment.
# The token stays server-side and is never shipped to the browser.
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TMDB_BASE = "https://api.themoviedb.org/3"

def tmdb_get(path: str, params: dict):
    token = os.environ.get("TMDB_READ_ACCESS_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="Indian movie catalogue needs TMDB_READ_ACCESS_TOKEN. Add a free TMDB read access token to your deployment environment.")
    url = f"{TMDB_BASE}{path}?{urlencode(params)}"
    req = Request(url, headers={"Authorization": f"Bearer {token}", "accept": "application/json", "User-Agent": "CineMatch/1.0"})
    try:
        with urlopen(req, timeout=12) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        log.warning("TMDB returned HTTP %s", exc.code)
        raise HTTPException(status_code=502, detail="The movie catalogue service returned an error. Please try again shortly.")
    except (URLError, TimeoutError, json.JSONDecodeError):
        raise HTTPException(status_code=502, detail="Could not reach the movie catalogue. Please try again shortly.")

def normalize_tmdb_movie(m: dict):
    date = m.get("release_date") or ""
    return {
        "id": m.get("id"),
        "title": m.get("title") or m.get("original_title") or "Untitled",
        "original_title": m.get("original_title") or "",
        "overview": m.get("overview") or "No synopsis available yet.",
        "year": date[:4] if len(date) >= 4 else "—",
        "rating": round(float(m.get("vote_average") or 0), 1),
        "votes": int(m.get("vote_count") or 0),
        "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
        "backdrop": f"https://image.tmdb.org/t/p/w1280{m['backdrop_path']}" if m.get("backdrop_path") else None,
        "language": m.get("original_language") or "",
        "genres": m.get("genre_ids") or [],
        "source": "TMDB",
    }

@app.get("/api/catalog/indian")
def indian_catalog(page: int = Query(1, ge=1, le=500), language: str | None = None, sort: str = "popularity.desc"):
    """Paginated Indian cinema catalogue; filter by original language (hi, ta, te, ml, kn, bn, mr, pa, gu, ur)."""
    allowed_sorts = {"popularity.desc", "vote_average.desc", "primary_release_date.desc", "primary_release_date.asc"}
    if sort not in allowed_sorts:
        sort = "popularity.desc"
    params = {"include_adult": "false", "include_video": "false", "language": "en-US",
              "page": page, "sort_by": sort, "with_origin_country": "IN"}
    if language:
        valid_languages = {"hi", "ta", "te", "ml", "kn", "bn", "mr", "pa", "gu", "ur", "as", "or", "en"}
        if language not in valid_languages:
            raise HTTPException(status_code=400, detail="Unsupported Indian cinema language filter.")
        params["with_original_language"] = language
    data = tmdb_get("/discover/movie", params)
    return {"page": data.get("page", page), "total_pages": data.get("total_pages", 0),
            "total_results": data.get("total_results", 0),
            "results": [normalize_tmdb_movie(m) for m in data.get("results", [])]}

@app.get("/api/catalog/hollywood")
def hollywood_catalog(page: int = Query(1, ge=1, le=500), sort: str = "popularity.desc"):
    """Browse English-language US films (Hollywood and other US cinema)."""
    allowed_sorts = {"popularity.desc", "vote_average.desc", "primary_release_date.desc", "primary_release_date.asc"}
    if sort not in allowed_sorts:
        sort = "popularity.desc"
    params = {"include_adult": "false", "include_video": "false", "language": "en-US",
              "page": page, "sort_by": sort, "with_origin_country": "US",
              "with_original_language": "en"}
    data = tmdb_get("/discover/movie", params)
    return {"page": data.get("page", page), "total_pages": data.get("total_pages", 0),
            "total_results": data.get("total_results", 0),
            "results": [normalize_tmdb_movie(m) for m in data.get("results", [])]}

@app.get("/api/catalog/search")
def catalog_search(q: str = Query(..., min_length=1, max_length=100), page: int = Query(1, ge=1, le=500), indian_only: bool = False):
    """Search movie titles through TMDB; restrict to Indian-origin movies by default."""
    params = {"query": q.strip(), "include_adult": "false", "language": "en-US", "page": page}
    if indian_only:
        params["with_origin_country"] = "IN"
    data = tmdb_get("/search/movie", params)
    results = data.get("results", [])
    if indian_only:
        # Search/movie does not accept the origin-country filter; use Indian-language
        # results here, while the dedicated Discover endpoint filters country=IN.
        indian_languages = {"hi", "ta", "te", "ml", "kn", "bn", "mr", "pa", "gu", "ur", "as", "or"}
        results = [m for m in results if m.get("original_language") in indian_languages]
    return {"page": data.get("page", page), "total_pages": data.get("total_pages", 0),
            "total_results": len(results) if indian_only else data.get("total_results", 0),
            "results": [normalize_tmdb_movie(m) for m in results]}

@app.get("/api/catalog/status")
def catalog_status():
    return {"tmdb_configured": bool(os.environ.get("TMDB_READ_ACCESS_TOKEN", "").strip()),
            "message": "Indian movie catalogue is ready." if os.environ.get("TMDB_READ_ACCESS_TOKEN", "").strip()
                       else "Add TMDB_READ_ACCESS_TOKEN to your hosting environment to enable the full Indian movie catalogue."}
