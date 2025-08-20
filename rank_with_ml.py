# rank_with_ml.py  -> uses saved model to output Fit %, gaps, reports
import os, re, json, joblib, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

roles = pd.read_csv("data/roles.csv")
resumes = pd.read_csv("data/resumes.csv")
resumes["clean"] = resumes["text"].str.lower().fillna("")
skills_cat = pd.read_csv("data/skills_catalog.csv")

print("\nAvailable roles:")
for i, r in roles.iterrows(): print(f"{i}. {r['role']}")
choice = input("\nEnter role number (blank = 0): ").strip()
idx = int(choice) if choice.isdigit() and int(choice) < len(roles) else 0

row = roles.iloc[idx]
role = row["role"]; jd_text = row["jd_text"].lower()
must = [s.strip() for s in row["must_have_skills"].split("|")]
nice = [s.strip() for s in row["nice_to_have_skills"].split("|")]

safe = re.sub(r"[^A-Za-z0-9_-]+","_", role)
vec = joblib.load(f"models/tfidf_{safe}.joblib")
clf = joblib.load(f"models/model_{safe}.joblib")

# similarity
X = vec.transform(list(resumes["clean"]) + [jd_text])
X_res, X_jd = X[:-1], X[-1]
sims = cosine_similarity(X_res, X_jd).ravel()

# ML prob
ml_prob = clf.predict_proba(X_res)[:,1]

# rule score with penalty
min_possible = -3*len(must); max_possible = 2*len(must) + 1*len(nice)
def rule_found(txt:str):
    t = txt.lower()
    found = set()
    for sk in set(must+nice):
        if sk in t: found.add(sk)
    miss = [m for m in must if m not in found]
    score = sum(2 for m in must if m in found) + sum(1 for n in nice if n in found) - 3*len(miss)
    norm = (score - min_possible) / (max_possible - min_possible) if max_possible>min_possible else 0
    return found, miss, norm

# training map
train_map = {}
if os.path.exists("data/training_map.csv"):
    dfm = pd.read_csv("data/training_map.csv")
    for _, r in dfm.iterrows(): train_map[r["skill"].strip().lower()] = (r["module_name"], int(r["duration_weeks"]))
else:
    train_map = {}

os.makedirs("outputs", exist_ok=True)
rows, reports = [], []
for i, r in resumes.iterrows():
    found, miss, rule_norm = rule_found(r["clean"])
    final = 0.5*ml_prob[i] + 0.3*sims[i] + 0.2*rule_norm
    fit_pct = int(round(100*final))
    sugg = [{"skill":m, "module":train_map.get(m,("",0))[0], "weeks":train_map.get(m,("",0))[1]} for m in miss]
    rows.append({
        "emp_id": r["emp_id"],
        "fit_pct": fit_pct,
        "ml_prob": round(float(ml_prob[i]),3),
        "similarity": round(float(sims[i]),3),
        "rule_norm": round(rule_norm,3),
        "matched": ", ".join(sorted(found)) if found else "-",
        "missing": "; ".join(miss) if miss else "-"
    })
    reports.append({
        "emp_id": r["emp_id"],
        "role": role,
        "fit_pct": fit_pct,
        "matched_skills": sorted(list(found)),
        "missing_skills": miss,
        "suggested_training": sugg
    })

out = pd.DataFrame(rows).sort_values("fit_pct", ascending=False)
print(f"\nRanking with ML for: {role}\n")
print(out.head(25).to_string(index=False))
out.to_csv(f"outputs/ranked_{safe}.csv", index=False)

for rep in reports:
    with open(f"outputs/skill_gap_{rep['emp_id']}.json","w",encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

print(f"\nSaved -> outputs/ranked_{safe}.csv + per-employee JSON in outputs/")
