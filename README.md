# skillgapanalyzer
Skill Gap Analyzer is an AI + ML tool that compares your skills with industry roles (Data Analyst, AI/ML Engineer, Data Scientist) and highlights the gaps. It also provides a personalized training plan to guide your learning journey.

Skill Gap Analyzer (Hybrid NLP + ML)

I recently developed a project called Skill Gap Analyzer, a free tool that compares candidate skills with industry roles (Data Analyst, AI/ML Engineer, Data Scientist).
It detects missing skills, highlights matched ones, and even suggests a personalized learning plan.

🔧 Tech Stack & Tools Used:

Python (core programming)

Streamlit (for interactive web app / deployment)

Pandas (data handling, CSV processing)

Scikit-learn (sklearn) → TF-IDF Vectorizer, Cosine Similarity, ML Models

NLP Techniques (text cleaning, preprocessing, rules + similarity)

PDFPlumber & Python-docx (for reading resumes in PDF/DOCX format)

Joblib (for saving/loading trained ML models)

CSV Data → roles.csv, skills_catalog.csv, resumes.csv, training_map.csv

⚙️ How it works:

Upload resume (PDF/DOCX/TXT) OR use sample data.

Select a target role (e.g., Data Analyst, ML Engineer).

App extracts skills → cleans & compares with required role skills.

Uses Hybrid Scoring:

TF-IDF similarity

Machine Learning prediction

Rule-based matching

Generates a fit percentage, missing skills, and a learning plan.

Results can be downloaded as CSV.

 Key Features:

Role-based skill requirements

Resume upload & parsing

Instant skill gap detection

Learning plan suggestions

No login needed, free to use
## 🚀 Live Demo  
[Click here to try Skill Gap Analyzer](https://skillgapanalyzer-9xhgpamacglyp5dkw9jep.streamlit.app/)



