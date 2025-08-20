import os, re, json, joblib, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("outputs", exist_ok=True)

roles   = pd.read_csv("data/roles.csv")
resumes = pd.read_csv("data/resumes.csv")
resumes["clean"] = resumes["text"].str.lower().fillna("")

# (optional) synonyms for rules
syn_map = {}
if os.path.exists("data/skills_catalog.csv"):
    sc = pd.read_csv("data/skills_catalog.csv")
    for _, r in sc.iterrows():
        key = r["skill"].strip().lower()
        vals = [key]
        if isinstance(r.get("synonyms"), str):
            vals += [p.strip().lower() for p in r["synonyms"].split("|") if p.strip()]
        syn_map[key] = set(vals)

def phrases_for(skill):
    s = skill.strip().lower()
    return syn_map.get(s, {s})

def rule_found(txt, must, nice):
    t = txt.lower()
    found=set()
    for sk in set(must+nice):
        if any(p and p in t for p in phrases_for(sk)):
            found.add(sk)
    miss=[m for m in must if m not in found]
    score = sum(2 for m in must if m in found) + sum(1 for n in nice if n in found) - 3*len(miss)
    return found, miss, score

summary_rows = []

for _, row in roles.iterrows():
    role = row["role"]
    jd   = row["jd_text"].lower()
    must = [s.strip() for s in str(row["must_have_skills"]).split("|") if s.strip()]
    nice = [s.strip() for s in str(row["nice_to_have_skills"]).split("|") if s.strip()]

    safe = re.sub(r"[^A-Za-z0-9_-]+","_", role)
    tfidf_path = f"models/tfidf_{safe}.joblib"
    model_path = f"models/model_{safe}.joblib"

    if not (os.path.exists(tfidf_path) and os.path.exists(model_path)):
        print(f"[skip] Model missing for role: {role}")
        continue

    vec = joblib.load(tfidf_path)
    clf = joblib.load(model_path)

    X = vec.transform(list(resumes["clean"]) + [jd])
    X_res, X_jd = X[:-1], X[-1]
    sims = cosine_similarity(X_res, X_jd).ravel()
    ml_prob = clf.predict_proba(X_res)[:,1]

    min_possible = -3*len(must)
    max_possible =  2*len(must) + 1*len(nice)
    out_rows = []

    for i, r in resumes.iterrows():
        found, miss, raw = rule_found(r["clean"], must, nice)
        rule_norm = (raw - min_possible) / (max_possible - min_possible) if max_possible>min_possible else 0.0
        final = 0.5*ml_prob[i] + 0.3*sims[i] + 0.2*rule_norm
        fit_pct = int(round(100*final))
        out_rows.append({
            "emp_id": r["emp_id"],
            "fit_pct": fit_pct,
            "ml_prob": round(float(ml_prob[i]),3),
            "similarity": round(float(sims[i]),3),
            "rule_norm": round(rule_norm,3),
            "matched": ", ".join(sorted(found)) if found else "-",
            "missing": "; ".join(miss) if miss else "-"
        })

    df = pd.DataFrame(out_rows).sort_values("fit_pct", ascending=False)
    df.to_csv(f"outputs/ranked_{safe}.csv", index=False)
    print(f"[saved] outputs/ranked_{safe}.csv  ({len(df)} rows)")

    # collect best role per employee for global summary
    top_for_role = df[["emp_id","fit_pct"]].copy()
    top_for_role["role"] = role
    summary_rows.append(top_for_role)

# merged summary: best role per employee
if summary_rows:
    merged = pd.concat(summary_rows)
    best = merged.sort_values("fit_pct", ascending=False).groupby("emp_id", as_index=False).first()
    best.rename(columns={"role":"best_role","fit_pct":"best_fit_pct"}, inplace=True)
    best.to_csv("outputs/fit_summary_all_roles.csv", index=False)
    print("[saved] outputs/fit_summary_all_roles.csv")
else:
    print("No roles processed (models missing?).")
