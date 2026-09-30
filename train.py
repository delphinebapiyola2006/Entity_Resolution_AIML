"""Load -> block -> features -> train RF / LogReg / GradBoost -> validate (threshold) -> test -> artifacts for the UI."""
import json, time, numpy as np, joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from er.pipeline import read_tsv, block, build, load_gt, best_threshold, run, FEATS

def prep(split):
    d = f"data/{split}"; s1 = read_tsv(f"{d}/{split}_source1.tsv"); gt = load_gt(f"{d}/{split}_ground_truth.tsv"); X, y = [], []
    for k in (2, 3):
        s = read_tsv(f"{d}/{split}_source{k}.tsv"); p = block(s1, s); X.append(build(s1, s, p))
        y.append(np.array([(s1.id[i], s.id[j]) in gt for i, j in p], int))
        print(f"[{split}] S1-S{k}: {len(p):,} candidates, blocking recall {y[-1].sum()/max(1,sum(1 for a,b in gt if b.startswith(f'S{k}'))):.3f}")
    return np.vstack(X), np.concatenate(y)

def pm(y, p, t):
    pr = p >= t; tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum()); fn = int(((~pr) & (y == 1)).sum()); tn = int(((~pr) & (y == 0)).sum())
    P = tp/max(1, tp+fp); R = tp/max(1, tp+fn); f = lambda b: (1+b*b)*P*R/max(1e-9, b*b*P+R)
    return dict(precision=P, recall=R, f1=f(1), f05=f(.5), tp=tp, fp=fp, fn=fn, tn=tn)

def stats(split, src):
    import pandas as pd
    df = pd.read_csv(f"data/{split}/{split}_source{src}.tsv", sep="\t", dtype=str, keep_default_na=False)
    return dict(n=len(df), missing={c: round(float((df[c].str.strip() == "").mean()*100), 2) for c in ("name", "address", "country")},
                unique_names=int(df.name.str.lower().nunique()), avg_name_len=round(float(df.name.str.len().mean()), 1),
                countries=df.country.value_counts().to_dict())

if __name__ == "__main__":
    t0 = time.time(); Xt, yt = prep("train"); Xv, yv = prep("validation"); Xs, ys = prep("test")
    rng = np.random.RandomState(0); keep = (yt == 1) | (rng.rand(len(yt)) < min(1, 250000/max(1, (yt == 0).sum())))
    Xt, yt = Xt[keep], yt[keep]; print(f"train pairs {len(yt):,} (pos {yt.sum():,})")
    models = {"Random Forest": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=0),
              "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=3, max_iter=2000, class_weight="balanced")),
              "Gradient Boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=.08, random_state=0)}
    M = {}; best = None
    for n, m in models.items():
        m.fit(Xt, yt); pv = m.predict_proba(Xv)[:, 1]; thr, vf = best_threshold(yv, pv); ps = m.predict_proba(Xs)[:, 1]
        fpr, tpr, _ = roc_curve(ys, ps); ix = np.linspace(0, len(fpr)-1, 40).astype(int)
        M[n] = dict(threshold=thr, val_f05=vf, test=pm(ys, ps, thr), auc=float(roc_auc_score(ys, ps)), ap=float(average_precision_score(ys, ps)),
                    roc=[[float(fpr[i]), float(tpr[i])] for i in ix], curve=[dict(t=round(float(t), 2), **pm(ys, ps, t)) for t in np.arange(.05, .96, .05)])
        if n == "Random Forest": M[n]["importance"] = dict(zip(FEATS, map(float, m.feature_importances_)))
        if n == "Logistic Regression":
            c = np.abs(m[-1].coef_[0]); M[n]["importance"] = dict(zip(FEATS, map(float, c/c.sum())))
        joblib.dump({"model": m, "thr": thr, "val_f05": vf, "name": n}, f"models/{n.lower().replace(' ', '_')}.joblib")
        print(f"{n:20s} val F0.5 {vf:.4f} | test P {M[n]['test']['precision']:.4f} R {M[n]['test']['recall']:.4f} F0.5 {M[n]['test']['f05']:.4f} AUC {M[n]['auc']:.4f}")
        if best is None or vf > M[best]["val_f05"]: best = n
    joblib.dump(joblib.load(f"models/{best.lower().replace(' ', '_')}.joblib"), "models/model.joblib")
    T = "data/test/test_"; r = run(T+"source1.tsv", T+"source2.tsv", T+"source3.tsv", gt_path=T+"ground_truth.tsv")
    json.dump(r, open("outputs/test_result.json", "w"))
    json.dump(dict(champion=best, models=M, record_level=r["stats"], features=FEATS, seconds=round(time.time()-t0)), open("outputs/metrics.json", "w"))
    ds = {"splits": {s: {k: stats(s, k) for k in (1, 2, 3)} for s in ("train", "validation", "test")}, "gt": {}}
    for s in ds["splits"]:
        c = Counter()
        for ln in open(f"data/{s}/{s}_ground_truth.tsv").read().splitlines()[1:]:
            ids = [x for x in ln.split("\t")[1].split(",") if x]; c[("2" in "".join(i[1] for i in ids)) * 1 + ("3" in "".join(i[1] for i in ids)) * 2] += 1
        ds["gt"][s] = dict(none=c[0], s2=c[1], s3=c[2], both=c[3])
    json.dump(ds, open("outputs/dataset_stats.json", "w")); print("CHAMPION:", best, "| TEST (record level):", r["stats"], f"| {time.time()-t0:.0f}s")
