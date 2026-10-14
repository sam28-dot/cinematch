# CineMatch (backend + ML core)

**Status:** backend API, ML recommenders, premium responsive web UI, About section, and TMDB-backed Indian cinema catalogue integration are implemented. The catalogue requires a free TMDB Read Access Token configured as `TMDB_READ_ACCESS_TOKEN` in the deployment environment.

## Run
    pip install -r requirements.txt
    # optional real data: download ml-100k from https://grouplens.org/datasets/movielens/ and unzip to data/raw/ml-100k/
    python -m ml.pipeline        # ETL -> data/processed, reports/data_quality.json (falls back to a labelled SYNTHETIC sample)
    python -m ml.evaluate        # -> reports/evaluation.json
    uvicorn backend.main:app --reload   # docs at http://localhost:8000/docs
    python -m pytest -q tests

## Algorithms
- **Popularity:** WR = v/(v+m)·R + m/(v+m)·C (m = 75th-percentile vote count, C = global mean), so one 5-star vote cannot top the list.
- **Content:** TF-IDF over genres, cosine similarity; user profile = mean vector of movies rated >= 4, or onboarding genres for cold start.
- **Collaborative:** item-item cosine kNN on the rating matrix (dense; fine for 100K, not for 1M+). A stand-in for Spark ALS, which is not implemented.
- **Hybrid:** min-max normalised weighted sum (content .4, CF .4, popularity .2; untuned), per-genre cap for diversity, watched/disliked removed, explanation = strongest contributing component.

## Indian cinema catalogue and UI
- The premium responsive UI lives in `frontend/index.html` and includes Discover, language filters, Indian cinema, a browser-based watchlist, and an About CineMatch section.
- `/api/catalog/indian` uses TMDB Discover with `with_origin_country=IN`; language filters include Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Marathi, Punjabi, Gujarati, Urdu, Assamese, Odia and English-language Indian films.
- `/api/catalog/search` searches Indian-origin films. Both endpoints use `TMDB_READ_ACCESS_TOKEN` server-side, never in browser code.
- To enable it, create a TMDB account, request a free API Read Access Token, then add `TMDB_READ_ACCESS_TOKEN` in your Render (or other host) environment variables and redeploy. Without this variable, the local recommendation engine still works but the full live catalogue is disabled.
- TMDB data and poster images are subject to TMDB's terms and attribution requirements. This is a searchable, paginated catalogue, not a downloaded static copy of every Indian film.

## Limits
MovieLens 100K has no language, runtime, cast or tags. `/api/providers` does not claim a film is free to watch unless verified provider data is configured. Data: MovieLens (GroupLens Research). TMDB attribution is required for TMDB data/images.

## Deploy
Single service: the API also serves a minimal UI at `/` (`frontend/index.html`, plain HTML/JS; the planned React app is not built).
- **Docker:** `docker build -t cinematch . && docker run -p 8000:8000 cinematch` then open http://localhost:8000
- **Render:** push the repo and create a Blueprint from `render.yaml` (free plan; check Render's current terms).
- **Data:** the image builds with whatever is in `data/raw/ml-100k/`; if absent it uses the labelled synthetic sample. Do not commit large raw data; copy it in before building if you want real MovieLens.
- **Persistence:** SQLite lists and ratings live in `data/cinematch.db`. Free hosts typically have ephemeral disks, so they reset on redeploy/restart. Set `CINEMATCH_DB` to a persistent path, or move to PostgreSQL, for durability.
- **CORS:** set `CINEMATCH_CORS` (comma-separated) if a separate frontend origin calls the API.
- There are no user accounts; identity is an anonymous browser id, so this is not secure multi-user auth.
- Tests: `pip install -r requirements-dev.txt && python -m pytest -q tests`
