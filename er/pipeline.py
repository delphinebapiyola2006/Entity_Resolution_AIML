"""Business Entity Resolution: normalize -> block -> features -> ML -> one-to-one assignment -> TSV."""
import csv, itertools, json, os
from collections import defaultdict
import numpy as np, pandas as pd, joblib
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.ensemble import HistGradientBoostingClassifier
from .preprocess import norm_name, norm_addr, norm_country

FEATS = ["jw_name","tset_name","tsort_name","partial_name","tri_name","first_tok","first4","name_len_ratio",
         "jw_addr","tset_addr","partial_addr","addr_jacc","num_match","num_conflict","same_country","name_addr_combo"]
MAXBLOCK = 60

def read_tsv(p):
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*ks):
        for k in ks:
            for lc, c in cols.items():
                if k in lc: return c
        return None
    idc = pick("entity_id","id") or df.columns[0]
    nc = pick("name","company","business") or df.columns[1]
    ac = pick("address","addr","street","location")
    cc = pick("country")
    out = pd.DataFrame({"id": df[idc], "name": df[nc], "address": df[ac] if ac else "", "country": df[cc] if cc else ""})
    out["n"] = out["name"].map(norm_name); out["a"] = out["address"].map(norm_addr); out["c"] = out["country"].map(norm_country)
    return out.reset_index(drop=True)

def _keys(r):
    t = r.n.split(); ks = set()
    if not t: return ks
    ks.add("f4:" + t[0][:4]); ks.add("f1:" + t[0])
    if len(t) > 1: ks.add("t2:" + "".join(sorted(t[:2]))); ks.add("l:" + t[-1] + t[0][:2])
    for w in t:
        if len(w) > 4: ks.add("w:" + w)
    nums = [w for w in r.a.split() if w.isdigit()]
    if nums and t: ks.add("na:" + nums[0] + t[0][:3])
    return ks

def block(s1, s2):
    idx = defaultdict(list)
    for j, r in enumerate(s2.itertuples()):
        for k in _keys(r): idx[k].append(j)
    pairs = set()
    for i, r in enumerate(s1.itertuples()):
        for k in _keys(r):
            b = idx.get(k, [])
            if len(b) <= MAXBLOCK:
                pairs.update((i, j) for j in b)
    return sorted(pairs)

def _tri(s):
    s = f"  {s} "; return {s[i:i+3] for i in range(len(s)-2)}
def feat(a, b):
    ta, tb = a.n.split(), b.n.split(); A, B = set(a.a.split()), set(b.a.split())
    na = {w for w in A if w.isdigit()}; nb = {w for w in B if w.isdigit()}
    tA, tB = _tri(a.n), _tri(b.n)
    f = [JaroWinkler.similarity(a.n, b.n), fuzz.token_set_ratio(a.n, b.n)/100, fuzz.token_sort_ratio(a.n, b.n)/100,
         fuzz.partial_ratio(a.n, b.n)/100, len(tA & tB)/max(1, len(tA | tB)),
         float(bool(ta and tb and ta[0] == tb[0])), float(bool(ta and tb and ta[0][:4] == tb[0][:4])),
         min(len(a.n), len(b.n))/max(1, max(len(a.n), len(b.n))),
         JaroWinkler.similarity(a.a, b.a), fuzz.token_set_ratio(a.a, b.a)/100, fuzz.partial_ratio(a.a, b.a)/100,
         len(A & B)/max(1, len(A | B)), float(bool(na and nb and na & nb)), float(bool(na and nb and not na & nb)),
         float(a.c == b.c or not a.c or not b.c), 0.0]
    f[-1] = f[1] * (0.5 + 0.5*f[9])
    return f

def _bc(A, B, c): return np.array([feat(A[i], B[j]) for i, j in c])
def build(s1, s2, pairs):
    if not pairs: return np.zeros((0, len(FEATS)))
    from joblib import Parallel, delayed
    A, B = list(s1.itertuples()), list(s2.itertuples())
    return np.vstack(Parallel(n_jobs=-1)(delayed(_bc)(A, B, pairs[k:k+20000]) for k in range(0, len(pairs), 20000)))

def load_gt(p):
    gt = set()
    for r in csv.DictReader(open(p), delimiter="\t"):
        for m in filter(None, r["matched_entity_ids"].split(",")): gt.add((r["source1_entity_id"], m.strip()))
    return gt

def best_threshold(y, p, beta=0.5):
    best = (0.5, 0)
    for t in np.arange(0.2, 0.96, 0.01):
        pr = p >= t; tp = (pr & (y == 1)).sum(); fp = (pr & (y == 0)).sum(); fn = ((~pr) & (y == 1)).sum()
        P = tp/max(1, tp+fp); R = tp/max(1, tp+fn); F = (1+beta**2)*P*R/max(1e-9, beta**2*P+R)
        if F > best[1]: best = (float(t), float(F))
    return best

def train(d="data/train", prefix="train", out="models/model.joblib"):
    s1 = read_tsv(f"{d}/{prefix}_source1.tsv"); gt = load_gt(f"{d}/{prefix}_ground_truth.tsv")
    Xs, ys, groups = [], [], []
    for k in (2, 3):
        s2 = read_tsv(f"{d}/{prefix}_source{k}.tsv"); pairs = block(s1, s2)
        X = build(s1, s2, pairs); y = np.array([(s1.id[i], s2.id[j]) in gt for i, j in pairs], dtype=int)
        Xs.append(X); ys.append(y); groups += [s1.id[i] for i, _ in pairs]
        tot = sum(1 for a, b in gt if b.startswith("S%d" % k))
        print("S1-S%d: %d candidates, %d true, blocking recall %.3f" % (k, len(pairs), y.sum(), y.sum()/max(1, tot)))
    X = np.vstack(Xs); y = np.concatenate(ys); g = np.array(groups)
    ids = np.unique(g); rng = np.random.RandomState(0); rng.shuffle(ids); val = set(ids[:len(ids)//5])
    v = np.array([x in val for x in g])
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=0).fit(X[~v], y[~v])
    thr, f = best_threshold(y[v], m.predict_proba(X[v])[:, 1])
    print(f"validation F0.5={f:.4f} at threshold {thr:.2f}")
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=0).fit(X, y)
    joblib.dump({"model": m, "thr": thr, "val_f05": f}, out); return out

def explain(f):
    names = {"jw_name":"Name spelling","tset_name":"Shared name words","tri_name":"Name character overlap","jw_addr":"Address spelling",
             "addr_jacc":"Address words","num_match":"Street number","same_country":"Country"}
    idx = {k: FEATS.index(k) for k in names}
    return [{"signal": names[k], "score": round(float(f[i]), 2)} for k, i in idx.items()]

def run(s1p, s2p, s3p, model="models/model.joblib", gt_path=None, outdir="outputs", thr=None):
    bundle = joblib.load(model); m = bundle["model"]; thr = thr if thr is not None else bundle["thr"]
    s1 = read_tsv(s1p); srcs = {2: read_tsv(s2p), 3: read_tsv(s3p)}
    matches = defaultdict(list); cand_rows = []; detail = defaultdict(list); ncand = 0
    for k, s in srcs.items():
        pairs = block(s1, s); ncand += len(pairs)
        if not pairs: continue
        X = build(s1, s, pairs); p = m.predict_proba(X)[:, 1]
        for (i, j), pr in zip(pairs, p): cand_rows.append((s1.id[i], s.id[j], round(float(pr), 4)))
        used1, used2 = set(), set()                     # one-to-one greedy assignment per source
        for n in np.argsort(-p):
            if p[n] < thr: break
            i, j = pairs[n]
            if i in used1 or j in used2: continue
            used1.add(i); used2.add(j); matches[i].append(s.id[j])
            r = s.iloc[j]
            detail[i].append({"id": r.id, "source": k, "conf": round(float(p[n]), 4), "name": r["name"], "address": r.address,
                              "country": r.country, "signals": explain(X[n])})
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/matching_results.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["source1_entity_id", "matched_entity_ids"])
        for i in range(len(s1)): w.writerow([s1.id[i], ",".join(matches.get(i, []))])
    with open(f"{outdir}/candidate_pairs.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["source1_entity_id", "candidate_entity_id", "score"]); w.writerows(cand_rows)
    rows = [{"id": s1.id[i], "name": s1.name[i], "address": s1.address[i], "country": s1.country[i],
             "matches": sorted(detail.get(i, []), key=lambda x: x["source"]),
             "conf": round(float(np.mean([d["conf"] for d in detail[i]])), 4) if i in detail else 0} for i in range(len(s1))]
    stats = {"total_s1": len(s1), "with_matches": sum(1 for r in rows if r["matches"]), "singletons": sum(1 for r in rows if not r["matches"]),
             "candidates": ncand, "pairs_matched": sum(len(r["matches"]) for r in rows), "threshold": round(thr, 2),
             "reduction": round(100*(1 - ncand/max(1, len(s1)*(len(srcs[2])+len(srcs[3])))), 2)}
    if gt_path and os.path.exists(gt_path):
        gt = load_gt(gt_path); pred = {(s1.id[i], d["id"]) for i in detail for d in detail[i]}
        tp = len(pred & gt); P = tp/max(1, len(pred)); R = tp/max(1, len(gt))
        stats.update({"precision": round(P, 4), "recall": round(R, 4), "f05": round(1.25*P*R/max(1e-9, .25*P+R), 4)})
    return {"stats": stats, "rows": rows}
