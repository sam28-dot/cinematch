"""Popularity (Bayesian weighted rating), TF-IDF content, item-kNN CF and hybrid ranking + offline evaluation.
NOTE: item-kNN CF is a stand-in; Spark ALS is NOT implemented yet."""
import math
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def mm(x):
    x = np.asarray(x, float); r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)

def popularity_stats(ratings):
    """WR = v/(v+m)*R + m/(v+m)*C ; m = 75th pct of vote counts, C = global mean."""
    g = ratings.groupby("movie_id").rating.agg(votes="count", avg="mean").reset_index()
    C, m = ratings.rating.mean(), g.votes.quantile(.75)
    g["wr"] = g.votes / (g.votes + m) * g.avg + m / (g.votes + m) * C
    return g, C

class Recommender:
    WEIGHTS = {"content": .4, "cf": .4, "pop": .2}

    def __init__(self, movies, ratings, weights=None):
        self.movies = movies.reset_index(drop=True); self.w = weights or self.WEIGHTS
        self.idx = {m: i for i, m in enumerate(self.movies.movie_id)}
        st, C = popularity_stats(ratings)
        self.stats = self.movies.merge(st, on="movie_id", how="left")
        self.stats["votes"] = self.stats.votes.fillna(0).astype(int); self.stats["wr"] = self.stats.wr.fillna(C)
        self.pop = self.stats.wr.to_numpy()
        txt = self.movies.genres.fillna("").str.replace(" ", "_").str.replace("|", " ")
        self.vec = TfidfVectorizer(token_pattern=r"[^ ]+").fit(txt); self.X = self.vec.transform(txt)
        R = np.zeros((ratings.user_id.nunique(), len(self.movies)))
        uid = {u: i for i, u in enumerate(ratings.user_id.unique())}
        for u, m, r in ratings[["user_id", "movie_id", "rating"]].itertuples(index=False):
            if m in self.idx: R[uid[u], self.idx[m]] = r
        Rn = R / np.maximum(np.linalg.norm(R, axis=0), 1e-9)
        self.cf = Rn.T @ Rn; np.fill_diagonal(self.cf, 0)

    def similar(self, movie_id, k=10):
        i = self.idx[movie_id]; s = cosine_similarity(self.X[i], self.X)[0]; s[i] = -1
        top = np.argsort(-s, kind="stable")[:k]
        return [{"movie_id": int(self.movies.movie_id[j]), "score": float(s[j]),
                 "reason": "Similar genres to the selected movie"} for j in top if s[j] > 0]

    def components(self, user_ratings, genres=None):
        liked = [self.idx[m] for m, r in user_ratings.items() if r >= 4 and m in self.idx]
        if liked: prof, why = np.asarray(self.X[liked].mean(0)), "Similar genres to movies you rated highly"
        elif genres: prof, why = self.vec.transform([" ".join(g.replace(" ", "_") for g in genres)]), "Matches your preferred genres"
        else: prof, why = None, ""
        r = np.zeros(len(self.movies))
        for m, v in user_ratings.items():
            if m in self.idx: r[self.idx[m]] = v
        return {"content": cosine_similarity(prof, self.X)[0] if prof is not None else np.zeros(len(r)),
                "cf": r @ self.cf, "pop": self.pop}, {"content": why, "cf": "Often rated highly by people who liked films you rated highly",
                "pop": "Highly rated with a solid number of votes (weighted rating)"}

    def rank(self, comps, whys, method="hybrid", k=10, exclude=(), diversify=True):
        w = self.w if method == "hybrid" else {method: 1.0}
        parts = {c: w.get(c, 0) * mm(v) for c, v in comps.items()}
        score = sum(parts.values()); cap, used, out = max(2, k // 3), {}, []
        for j in np.argsort(-score, kind="stable"):
            mid = int(self.movies.movie_id[j])
            if mid in exclude or score[j] <= 0: continue
            g = (self.movies.genres[j] or "").split("|")[0]
            if diversify and used.get(g, 0) >= cap: continue
            used[g] = used.get(g, 0) + 1
            best = max(parts, key=lambda c: parts[c][j])
            out.append({"movie_id": mid, "score": float(score[j]), "reason": whys[best] or whys["pop"]})
            if len(out) == k: break
        return out

    def recommend(self, user_ratings, genres=None, k=10, exclude=(), method="hybrid"):
        comps, whys = self.components(user_ratings, genres)
        return self.rank(comps, whys, method, k, set(exclude) | set(user_ratings))

def dcg(rels): return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

def evaluate(movies, ratings, k=10, thresh=4, max_users=300, seed=0):
    """Per-user temporal split (last 20% held out). Relevant = held-out rating >= thresh. Models see train only."""
    ratings = ratings.sort_values("timestamp"); parts = []
    for _, g in ratings.groupby("user_id"):
        if len(g) >= 5: n = max(1, int(len(g) * .2)); parts.append((g.iloc[:-n], g.iloc[-n:]))
    train = pd.concat([p[0] for p in parts]); rec = Recommender(movies, train)
    rng = np.random.default_rng(seed); pick = rng.permutation(len(parts))[:max_users]
    m = {x: {"p": [], "r": [], "n": [], "seen": set()} for x in ["popularity", "content", "cf", "hybrid"]}
    keymap = {"popularity": "pop", "content": "content", "cf": "cf", "hybrid": "hybrid"}
    for i in pick:
        tr, te = parts[i]; rel = set(te[te.rating >= thresh].movie_id)
        if not rel: continue
        comps, whys = rec.components(dict(zip(tr.movie_id, tr.rating)))
        for name in m:
            recs = [x["movie_id"] for x in rec.rank(comps, whys, keymap[name], k, set(tr.movie_id))]
            hits = [1 if x in rel else 0 for x in recs]
            m[name]["p"].append(sum(hits) / k); m[name]["r"].append(sum(hits) / len(rel))
            m[name]["n"].append(dcg(hits) / max(dcg([1] * min(len(rel), k)), 1e-9)); m[name]["seen"].update(recs)
    return {name: {f"precision@{k}": float(np.mean(v["p"])), f"recall@{k}": float(np.mean(v["r"])),
                   f"ndcg@{k}": float(np.mean(v["n"])), "catalog_coverage": len(v["seen"]) / len(movies)}
            for name, v in m.items() if v["p"]} | {"_meta": {"users_evaluated": len(m["hybrid"]["p"]), "k": k,
            "relevance_threshold": thresh, "split": "per-user temporal 80/20", "rmse_mae": "not computed (ranking models, no rating predictor)"}}
