# train_ml.py (v2)  — dynamic thresholds + fallback so 2 classes always
import os, re, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("models", exist_ok=True)

roles = pd.read_csv("data/roles.csv")
resumes = pd.read_csv("data/resumes.csv")
resumes["clean"] = resumes["text"].str.lower().fillna("")

print("\nAvailable roles:")
for i, r in roles.iterrows(): print(f"{i}. {r['role']}")
choice = input("\nTrain for role number (blank = 0): ").strip()
idx = int(choice) if choice.isdigit() and int(choice) < len(roles) else 0

row = roles.iloc[idx]
role_name = row["role"]
jd_text = row["jd_text"].lower()
must = [s.strip() for s in row["must_have_skills"].split("|")]

# TF-IDF on resumes only (no leakage)
vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
X_res = vec.fit_transform(resumes["clean"])
X_jd  = vec.transform([jd_text])
sims = cosine_similarity(X_res, X_jd).ravel()

# coverage of must-haves
def coverage(t):
    t = t.lower()
    return sum(1 for m in must if m in t) / max(1, len(must))

cov = resumes["clean"].apply(coverage).values

# ---- dynamic weak labels ----
sim_thr = max(0.12, float(np.percentile(sims, 70)))   # 70th percentile or 0.12
cov_thr = 0.5 if len(must) else 0.0                   # at least half must-haves
y = ((sims >= sim_thr) & (cov >= cov_thr)).astype(int)

# fallback: ensure both classes exist
pos = int(y.sum())
if pos == 0:
    top_k = max(10, int(0.1 * len(y)))
    y[np.argsort(sims)[-top_k:]] = 1
elif pos == len(y):
    bot_k = max(10, int(0.2 * len(y)))
    y[np.argsort(sims)[:bot_k]] = 0

print(f"Label counts -> pos: {int(y.sum())}, neg: {int((1-y).sum())}")

# train
Xtr, Xte, ytr, yte = train_test_split(X_res, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
print(f"F1 (weak labels): {f1_score(yte, pred):.3f}")

safe = re.sub(r"[^A-Za-z0-9_-]+","_", role_name)
joblib.dump(vec,  f"models/tfidf_{safe}.joblib")
joblib.dump(clf,  f"models/model_{safe}.joblib")
print(f"Saved -> models/tfidf_{safe}.joblib , models/model_{safe}.joblib")
