import os, tempfile
os.environ["CINEMATCH_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
import pandas as pd, pytest
from fastapi.testclient import TestClient
from ml.pipeline import make_sample
from ml.recs import Recommender, evaluate
from backend.main import app

@pytest.fixture(scope="module")
def rec():
    m, r = make_sample(); return Recommender(m, r)

def test_single_vote_not_top():
    m = pd.DataFrame({"movie_id": [1, 2, 3], "title": list("abc"), "year": [2000] * 3, "genres": ["Drama"] * 3})
    r = pd.DataFrame({"user_id": list(range(101)), "movie_id": [1] + [2] * 50 + [3] * 50, "rating": [5] + [4] * 50 + [2] * 50})
    assert Recommender(m, r).stats.sort_values("wr").iloc[-1].movie_id == 2

def test_similar_excludes_self(rec):
    assert 1 not in [x["movie_id"] for x in rec.similar(1, 10)]

def test_hybrid_excludes_seen_and_no_dupes(rec):
    out = rec.recommend({1: 5, 2: 5, 3: 1}, k=10); ids = [x["movie_id"] for x in out]
    assert not {1, 2, 3} & set(ids) and len(ids) == len(set(ids)) and all(x["reason"] for x in out)

def test_cold_start(rec):
    assert len(rec.recommend({}, genres=["Drama"], k=5)) == 5

def test_evaluation_metrics_in_range():
    m, r = make_sample(); res = evaluate(m, r, max_users=40)
    for n in ["popularity", "content", "cf", "hybrid"]: assert 0 <= res[n]["ndcg@10"] <= 1

def test_api():
    with TestClient(app) as c:
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/movies/999999").status_code == 404
        gs = c.get("/api/genres").json(); g = gs[0]
        res = c.get("/api/movies", params={"genres": g, "min_votes": 5}).json()["results"]
        assert res and all(g in x["genres"].split("|") and x["votes"] >= 5 for x in res)
        assert c.put("/api/users/u1/movies/1", json={"status": "watched", "rating": 5}).status_code == 200
        assert c.put("/api/users/u1/movies/1", json={"rating": 9}).status_code == 422
        assert c.get("/api/users/u1/movies", params={"status": "watched"}).json()[0]["rating"] == 5
        ids = [x["movie_id"] for x in c.get("/api/recommendations/user/u1").json()["results"]]
        assert 1 not in ids
        assert c.get("/api/providers/1").json()["providers"] == []

def test_index_served():
    with TestClient(app) as c:
        r = c.get("/"); assert r.status_code == 200 and "CineMatch" in r.text
