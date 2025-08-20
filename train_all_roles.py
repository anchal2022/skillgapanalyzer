# train_all_roles.py
import os, re, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs("models", exist_ok=True)
roles   = pd.read_csv("data/roles.csv")
resumes = pd.read_csv("data/resumes.csv")
resumes["clean"] = resumes["text"].str.lower().fillna("")

for _, row in roles.iterrows():
    role = row["role"]; jd = row["jd_text"].lower()
    must = [s.strip() for s in row["must_have_skills"].split("|")]

    vec  = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    Xres = vec.fit_transform(resumes["clean"])
    Xjd  = vec.transform([jd])
    sims = cosine_similarity(Xres, Xjd).ravel()

    def cov(t): return sum(1 for m in must if m in t)/max(1,len(must))
    cover = resumes["clean"].apply(cov).values

    sim_thr = max(0.12, float(np.percentile(sims, 70)))
    cov_thr = 0.5 if len(must) else 0.0
    y = ((sims>=sim_thr)&(cover>=cov_thr)).astype(int)

    # ensure two classes
    if y.sum()==0:
        y[np.argsort(sims)[-max(10,int(0.1*len(y))):]] = 1
    if y.sum()==len(y):
        y[np.argsort(sims)[:max(10,int(0.2*len(y)))] ] = 0

    Xtr,Xte,ytr,yte = train_test_split(Xres,y,test_size=0.2,random_state=42,stratify=y)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(Xtr,ytr)
    print(f"{role}: F1={f1_score(yte, clf.predict(Xte)):.3f}  pos={int(y.sum())} neg={int((1-y).sum())}")

    safe = re.sub(r"[^A-Za-z0-9_-]+","_", role)
    joblib.dump(vec, f"models/tfidf_{safe}.joblib")
    joblib.dump(clf, f"models/model_{safe}.joblib")
print("All done ✅")
