import json, pandas as pd
from ml.pipeline import PROC, REPORTS, process
from ml.recs import evaluate
if __name__ == "__main__":
    if not (PROC / "movies.csv").exists(): process()
    src = json.loads((REPORTS / "data_quality.json").read_text())
    res = evaluate(pd.read_csv(PROC / "movies.csv"), pd.read_csv(PROC / "ratings.csv"))
    res["_meta"].update(source=src["source"], is_sample=src["is_sample"])
    (REPORTS / "evaluation.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
