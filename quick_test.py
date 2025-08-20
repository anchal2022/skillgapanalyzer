# quick_test.py  (v2)  -> cleaning + synonyms + -3 penalty
import re, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

roles = pd.read_csv("data/roles.csv")
resumes = pd.read_csv("data/resumes.csv")
skills_cat = pd.read_csv("data/skills_catalog.csv")

# pick role interactively
print("\nAvailable roles:")
for i, r in roles.iterrows(): print(f"{i}. {r['role']}")
choice = input("\nEnter role number (blank = 0): ").strip()
idx = int(choice) if choice.isdigit() and int(choice) < len(roles) else 0
row = roles.iloc[idx]
role_name = row["role"]; jd_text = row["jd_text"]
must = [s.strip() for s in row["must_have_skills"].split("|")]
nice = [s.strip() for s in row["nice_to_have_skills"].split("|")]

# --- build synonyms map (skill -> set of phrases) ---
syn_map = {}
for _, r in skills_cat.iterrows():
    phrases = [r["skill"].strip().lower()]
    if isinstance(r["synonyms"], str):
        phrases += [p.strip().lower() for p in r["synonyms"].split("|") if p.strip()]
    syn_map[r["skill"].strip().lower()] = set(phrases)

def clean(t:str)->str:
    t = t or ""
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)      # urls
    t = re.sub(r"\S+@\S+", " ", t)                    # emails
    t = re.sub(r"\+?\d[\d\-\s]{6,}\d", " ", t)        # phones
    t = re.sub(r"[^a-z0-9\s]", " ", t)                # special chars
    t = re.sub(r"\s+", " ", t).strip()
    return t

resumes["clean"] = resumes["text"].apply(clean)
jd_clean = clean(jd_text)

def match_skills(txt:str):
    found = set()
    for sk, phrases in syn_map.items():
        if any(p and p in txt for p in phrases):  # simple contains
            found.add(sk)
    # years extraction (optional info)
    years = re.findall(r"(\d+)\s+years?", txt)
    years = max([int(y) for y in years], default=0)
    return found, years

# TF-IDF sim to JD (on cleaned text)
vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=3000)
X = vec.fit_transform(list(resumes["clean"]) + [jd_clean])
X_res, X_jd = X[:-1], X[-1]
sims = cosine_similarity(X_res, X_jd).ravel()

# rule score with -3 penalty for missing must-haves
min_possible = -3*len(must)
max_possible = 2*len(must) + 1*len(nice)
rows = []
for i, r in resumes.iterrows():
    found, years = match_skills(r["clean"])
    miss = [m for m in must if m not in found]
    score = sum(2 for m in must if m in found) + sum(1 for n in nice if n in found) - 3*len(miss)
    rule_norm = (score - min_possible) / (max_possible - min_possible) if max_possible>min_possible else 0.0
    final = 0.7*sims[i] + 0.3*rule_norm
    rows.append({
        "emp_id": r["emp_id"],
        "similarity": round(float(sims[i]),3),
        "rule_norm": round(rule_norm,3),
        "final_score": round(float(final),3),
        "matched_skills": ", ".join(sorted(found)) if found else "-",
        "missing_must_haves": "; ".join(miss) if miss else "-",
        "years_detected": years
    })

out = pd.DataFrame(rows).sort_values("final_score", ascending=False)
print(f"\nRanking for role: {role_name}\n")
print(out.head(25).to_string(index=False))  # top 25 shown
safe = re.sub(r"[^A-Za-z0-9_-]+","_", role_name)
out.to_csv(f"outputs_ranked_{safe}.csv", index=False)
print(f"\nSaved -> outputs_ranked_{safe}.csv")
