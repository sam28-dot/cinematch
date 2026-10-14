"""ETL: MovieLens 100K (data/raw/ml-100k) -> data/processed + data-quality report.
Falls back to a seeded SYNTHETIC sample when raw files are absent (labelled in the report)."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, REPORTS = ROOT / "data/raw/ml-100k", ROOT / "data/processed", ROOT / "reports"
GENRES = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

def load_raw():
    if not ((RAW / "u.data").exists() and (RAW / "u.item").exists()):
        return None
    r = pd.read_csv(RAW / "u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    m = pd.read_csv(RAW / "u.item", sep="|", encoding="latin-1", header=None,
                    usecols=[0, 1, 2, *range(5, 24)], names=["movie_id", "title", "release_date", *GENRES])
    m["year"] = pd.to_datetime(m.release_date, errors="coerce").dt.year.astype("Int64")
    m["genres"] = m[GENRES[1:]].apply(lambda row: "|".join(g for g, v in zip(GENRES[1:], row) if v == 1), axis=1)
    return m[["movie_id", "title", "year", "genres"]], r

def make_sample(n_movies=300, n_users=200, seed=42):
    rng = np.random.default_rng(seed)
    gl = GENRES[1:9]
    mg = [list(rng.choice(gl, rng.integers(1, 4), replace=False)) for _ in range(n_movies)]
    movies = pd.DataFrame({"movie_id": range(1, n_movies + 1),
                           "title": [f"Sample Film {i:03d}" for i in range(1, n_movies + 1)],
                           "year": rng.integers(1980, 2024, n_movies), "genres": ["|".join(g) for g in mg]})
    quality, rows = rng.normal(0, .5, n_movies), []
    for u in range(1, n_users + 1):
        fav = set(rng.choice(gl, 2, replace=False))
        for i in rng.choice(n_movies, rng.integers(30, 61), replace=False):
            s = 3 + 1.2 * len(fav & set(mg[i])) - .6 * (not fav & set(mg[i])) + quality[i] + rng.normal(0, .5)
            rows.append((u, i + 1, int(np.clip(round(s), 1, 5)), 880000000 + int(rng.integers(0, 1e7))))
    return movies, pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])

def process(force_sample=False):
    t0 = time.time()
    raw = None if force_sample else load_raw()
    source = "movielens-100k" if raw else "synthetic_sample"
    movies, ratings = raw or make_sample()
    n0 = len(ratings)
    ratings = ratings.drop_duplicates(["user_id", "movie_id"]); dups = n0 - len(ratings)
    ratings = ratings[ratings.rating.between(1, 5) & ratings.movie_id.isin(movies.movie_id)]
    PROC.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(exist_ok=True)
    movies.to_csv(PROC / "movies.csv", index=False); ratings.to_csv(PROC / "ratings.csv", index=False)
    report = {"source": source, "is_sample": source != "movielens-100k", "movies": len(movies),
              "ratings": len(ratings), "users": int(ratings.user_id.nunique()), "duplicates_removed": dups,
              "invalid_removed": n0 - dups - len(ratings),
              "missing_values": {c: int(v) for c, v in movies.isna().sum().items()},
              "rating_distribution": {int(k): int(v) for k, v in ratings.rating.value_counts().sort_index().items()},
              "etl_seconds": round(time.time() - t0, 3)}
    (REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    print(json.dumps(process(), indent=2))
