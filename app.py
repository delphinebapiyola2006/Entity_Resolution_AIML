import os, json, tempfile, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse, JSONResponse
from er.pipeline import run, read_tsv, block, build, explain
app = FastAPI(title="Entity Resolution Studio")
J = lambda n: json.load(open(f"outputs/{n}.json"))
T = "data/test/test_"; _c = {}
def cache(k, f):
    if k not in _c: _c[k] = f()
    return _c[k]
def srcs(): return cache("s", lambda: {k: read_tsv(f"{T}source{k}.tsv") for k in (1, 2, 3)})

@app.get("/api/overview")
def overview(): return {**J("dataset_stats"), "champion": J("metrics")["champion"], "record_level": J("metrics")["record_level"]}
@app.get("/api/metrics")
def metrics(): return J("metrics")
@app.get("/api/result")
def result(): return cache("r", lambda: J("test_result"))
@app.get("/api/unmatched")
def unmatched():
    def f():
        S = srcs(); R = J("test_result")["rows"]; hit = {m["id"] for r in R for m in r["matches"]}
        best = pd.read_csv("outputs/candidate_pairs.tsv", sep="\t").sort_values("score", ascending=False).drop_duplicates("source1_entity_id").set_index("source1_entity_id")
        nm = {i: n for k in (2, 3) for i, n in zip(S[k].id, S[k].name)}; out = {"s1": [], "s2": [], "s3": []}
        for r in R:
            if not r["matches"]:
                b = best.loc[r["id"]] if r["id"] in best.index else None
                out["s1"].append({**{k: r[k] for k in ("id", "name", "address", "country")}, "best": b.candidate_entity_id if b is not None else "", "best_name": nm.get(b.candidate_entity_id, "") if b is not None else "", "score": float(b.score) if b is not None else 0})
        for k in (2, 3): out[f"s{k}"] = [dict(id=a, name=b, address=c, country=d) for a, b, c, d in zip(S[k].id, S[k].name, S[k].address, S[k].country) if a not in hit]
        return out
    return cache("u", f)
@app.get("/api/explore/{split}/{src}")
def explore(split: str, src: int, q: str = "", page: int = 0):
    if split not in ("train", "validation", "test") or src not in (1, 2, 3): return JSONResponse({"error": "bad"}, 400)
    df = cache(f"e{split}{src}", lambda: pd.read_csv(f"data/{split}/{split}_source{src}.tsv", sep="\t", dtype=str, keep_default_na=False))
    if q: df = df[df.apply(lambda r: q.lower() in " ".join(r).lower(), axis=1)]
    return {"total": len(df), "rows": df.iloc[page*20:(page+1)*20].to_dict("records")}
@app.get("/api/files")
def files():
    o = []
    for root, _, fs in sorted(os.walk(".")):
        if any(x in root for x in ("__pycache__", ".git")): continue
        for f in sorted(fs):
            p = os.path.join(root, f); o.append({"path": p[2:], "kb": round(os.path.getsize(p)/1024, 1), "rows": sum(1 for _ in open(p, "rb"))-1 if f.endswith(".tsv") else None})
    return o
@app.post("/api/resolve")
def resolve(b: dict = Body(...)):
    q = pd.DataFrame([{"entity_id": "Q", "name": b.get("name", ""), "address": b.get("address", ""), "country": b.get("country", "")}]); q.to_csv("/tmp/q.tsv", sep="\t", index=False)
    q = read_tsv("/tmp/q.tsv"); m = joblib.load("models/model.joblib"); S = srcs(); out = []
    for k in (2, 3):
        s = S[k]; p = block(q, s)
        if not p: continue
        X = build(q, s, p); pr = m["model"].predict_proba(X)[:, 1]
        for n in np.argsort(-pr)[:4]:
            r = s.iloc[p[n][1]]; out.append(dict(id=r.id, source=k, name=r["name"], address=r.address, country=r.country, prob=round(float(pr[n]), 4), match=bool(pr[n] >= m["thr"]), signals=explain(X[n])))
    return sorted(out, key=lambda x: -x["prob"])[:8]
@app.post("/api/run")
async def api_run(s1: UploadFile = File(...), s2: UploadFile = File(...), s3: UploadFile = File(...)):
    d = tempfile.mkdtemp(); ps = []
    for n, f in zip("123", (s1, s2, s3)):
        p = f"{d}/s{n}.tsv"; open(p, "wb").write(await f.read()); ps.append(p)
    try:
        r = run(*ps, outdir=d); return {"stats": r["stats"], "rows": r["rows"][:200]}
    except Exception as e: return JSONResponse({"error": str(e)}, 400)
@app.get("/api/download/{name}")
def dl(name: str):
    if name not in ("matching_results.tsv", "candidate_pairs.tsv"): return JSONResponse({"error": "no"}, 404)
    return FileResponse(f"outputs/{name}", filename=name)
@app.get("/")
def index(): return FileResponse("web/index.html")
