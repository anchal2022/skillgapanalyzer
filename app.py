# app_streamlit.py  — simple UI for Skill Gap Analyzer
import os, re, io, json, joblib
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- helpers ----------
def clean(t: str) -> str:
    import re
    t = (t or "").lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\S+@\S+", " ", t)
    t = re.sub(r"\+?\d[\d\-\s]{6,}\d", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def read_file(file):
    name = file.name.lower()
    if name.endswith(".txt"):
        return file.name, file.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            txt = "\n".join([(p.extract_text() or "") for p in pdf.pages])
        return file.name, txt
    if name.endswith(".docx"):
        from docx import Document
        bio = io.BytesIO(file.read())
        doc = Document(bio)
        return file.name, "\n".join([p.text for p in doc.paragraphs])
    # default: try decode
    return file.name, file.read().decode("utf-8", errors="ignore")

def load_syn_map():
    syn = {}
    if os.path.exists("data/skills_catalog.csv"):
        df = pd.read_csv("data/skills_catalog.csv")
        for _, r in df.iterrows():
            key = str(r["skill"]).strip().lower()
            vals = [key]
            if isinstance(r.get("synonyms"), str):
                vals += [p.strip().lower() for p in r["synonyms"].split("|") if p.strip()]
            syn[key] = set(vals)
    return syn

def phrases_for(skill, syn_map):
    s = skill.strip().lower()
    return syn_map.get(s, {s})

def rule_found(txt, must, nice, syn_map):
    t = txt.lower()
    found=set()
    for sk in set(must+nice):
        if any(p and p in t for p in phrases_for(sk, syn_map)):
            found.add(sk)
    miss=[m for m in must if m not in found]
    score = sum(2 for m in must if m in found) + sum(1 for n in nice if n in found) - 3*len(miss)
    return found, miss, score

# ---------- data ----------
roles = pd.read_csv("data/roles.csv")
training_map = {}
if os.path.exists("data/training_map.csv"):
    dfm = pd.read_csv("data/training_map.csv")
    for _, r in dfm.iterrows():
        training_map[str(r["skill"]).strip().lower()] = (r["module_name"], int(r["duration_weeks"]))
syn_map = load_syn_map()

# ---------- UI ----------
st.set_page_config(page_title="Skill Gap Analyzer", layout="wide")
st.title("🧠 Skill Gap Analyzer (Hybrid NLP + ML)")

role_names = roles["role"].tolist()
role = st.selectbox("Choose role", role_names, index=0)
row = roles[roles["role"]==role].iloc[0]
jd_text_default = row["jd_text"]
must = [s.strip() for s in str(row["must_have_skills"]).split("|") if s.strip()]
nice = [s.strip() for s in str(row["nice_to_have_skills"]).split("|") if s.strip()]

st.markdown(f"**Must-have:** `{', '.join(must)}`  |  **Nice-to-have:** `{', '.join(nice)}`")
jd_text = st.text_area("Job description (you can edit)", value=jd_text_default, height=120)

st.subheader("Upload resumes")
files = st.file_uploader("PDF / DOCX / TXT (multiple allowed)", type=["pdf","docx","txt"], accept_multiple_files=True)
use_existing = st.checkbox("Or use existing data/resumes.csv", value=not files)

if st.button("Rank now"):
    # collect resumes
    resumes = []
    if files:
        for f in files:
            name, txt = read_file(f)
            resumes.append((os.path.splitext(name)[0], txt))
    if use_existing:
        df = pd.read_csv("data/resumes.csv")
        for _, r in df.iterrows():
            resumes.append((str(r["emp_id"]), str(r["text"])))
    if not resumes:
        st.warning("No resumes provided.")
        st.stop()

    # clean
    ids = [rid for rid,_ in resumes]
    texts_raw = [txt for _,txt in resumes]
    texts = [clean(t) for t in texts_raw]
    jd_clean = clean(jd_text)

    # try to load trained model for this role
    safe = re.sub(r"[^A-Za-z0-9_-]+","_", role)
    tfidf_path = f"models/tfidf_{safe}.joblib"
    model_path = f"models/model_{safe}.joblib"
    use_ml = os.path.exists(tfidf_path) and os.path.exists(model_path)

    if use_ml:
        vec = joblib.load(tfidf_path)
        clf = joblib.load(model_path)
        X = vec.transform(texts + [jd_clean])
        X_res, X_jd = X[:-1], X[-1]
        sims = cosine_similarity(X_res, X_jd).ravel()
        ml_prob = clf.predict_proba(X_res)[:,1]
    else:
        st.info("No trained model found for this role — using similarity + rules only.")
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=4000)
        X = vec.fit_transform(texts + [jd_clean])
        X_res, X_jd = X[:-1], X[-1]
        sims = cosine_similarity(X_res, X_jd).ravel()
        ml_prob = [0.0]*len(texts)

    min_possible = -3*len(must)
    max_possible =  2*len(must) + 1*len(nice)

    rows = []
    for i, txt in enumerate(texts):
        found, miss, raw = rule_found(txt, must, nice, syn_map)
        rule_norm = (raw - min_possible)/(max_possible-min_possible) if max_possible>min_possible else 0.0
        final = 0.5*(ml_prob[i]) + 0.3*(sims[i]) + 0.2*rule_norm
        fit_pct = int(round(100*final))
        sugg = [{"skill":m,
                 "module":training_map.get(m,("",""))[0],
                 "weeks":training_map.get(m,("",""))[1]} for m in miss]
        rows.append({
            "emp_id": ids[i],
            "fit_pct": fit_pct,
            "ml_prob": round(float(ml_prob[i]),3),
            "similarity": round(float(sims[i]),3),
            "rule_norm": round(rule_norm,3),
            "matched": ", ".join(sorted(found)) if found else "-",
            "missing": "; ".join(miss) if miss else "-",
            "suggested_training": json.dumps(sugg, ensure_ascii=False)
        })

    out = pd.DataFrame(rows).sort_values("fit_pct", ascending=False).reset_index(drop=True)
    st.success(f"Ranking ready for **{role}**")
    st.dataframe(out, use_container_width=True)

    # download
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name=f"ranked_{safe}.csv", mime="text/csv")
