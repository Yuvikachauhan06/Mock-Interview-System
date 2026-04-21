from flask import Flask, request, render_template
import pandas as pd
import pdfplumber
import pickle
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- ROLE WEIGHTS ----------------
ROLE_WEIGHTS = {
    "Software Engineer": {
        "DSA": 40, "Python": 10, "Java": 5, "C++": 5,
        "Operating Systems": 10, "Computer Networks": 10,
        "System Design": 10, "OOPS": 10
    },
    "Backend Developer": {
        "Backend Development": 35, "DBMS": 25, "System Design": 15,
        "Python": 5, "Java": 5, "Cloud & DevOps": 10, "Computer Networks": 5
    },
    "Frontend Developer": {
        "Web Development": 50, "System Design": 15, "OOPS": 10,
        "DBMS": 10, "DSA": 15
    },
    "Full-Stack Developer": {
        "Web Development": 30, "Backend Development": 25, "DBMS": 15,
        "System Design": 15, "Cloud & DevOps": 10, "DSA": 5
    },
    "Data Scientist": {
        "Data Science": 40, "AIML": 30, "Python": 15,
        "DBMS": 10, "OOPS": 5
    },
    "Machine Learning Engineer": {
        "AIML": 40, "Python": 20, "Data Science": 20,
        "System Design": 10, "Cloud & DevOps": 10
    },
    "Database Engineer": {
        "DBMS": 40, "Backend Development": 20, "Python": 10,
        "Java": 5, "System Design": 15, "Cloud & DevOps": 10
    },
    "Cybersecurity Engineer": {
        "Cybersecurity": 40, "Computer Networks": 25,
        "Operating Systems": 15, "System Design": 10, "OOPS": 10
    },
    "Cloud / DevOps Engineer": {
        "Cloud & DevOps": 40, "Backend Development": 20,
        "System Design": 15, "DBMS": 10, "Computer Networks": 10,
        "Cybersecurity": 5
    },
    "Core CS / Research": {
        "DSA": 40, "Operating Systems": 20, "Computer Networks": 20,
        "System Design": 10, "OOPS": 10, "C++": 5,
        "Java": 5, "Python": 5
    }
}

# ---------------- PDF TEXT ----------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.lower()

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    skills = []
    mapping = {
        "DSA": ["dsa", "data structures"],
        "Python": ["python"],
        "Java": ["java"],
        "C++": ["c++", "cpp"],
        "AIML": ["machine learning", "ai"],
        "Data Science": ["statistics", "pandas"],
        "DBMS": ["sql", "database", "dbms"],
        "Web Development": ["html", "css", "javascript", "react"],
        "Backend Development": ["django", "flask", "node", "api"],
        "Operating Systems": ["os", "process", "thread"],
        "Computer Networks": ["network", "tcp", "ip"],
        "System Design": ["scalability", "architecture"],
        "Cloud & DevOps": ["aws", "docker", "kubernetes"],
        "Cybersecurity": ["security", "encryption"],
        "OOPS": ["oops", "object oriented"]
    }
    for domain, keys in mapping.items():
        if any(k.lower() in text for k in keys):
            skills.append(domain)
    return skills

# ---------------- NORMALIZE ----------------
def normalize_weights(role_weights, skills):
    filtered = {k: v for k, v in role_weights.items() if k in skills}
    if not filtered:  # fallback
        total = sum(role_weights.values())
        return {k: v/total for k, v in role_weights.items()}
    total = sum(filtered.values())
    return {k: v/total for k, v in filtered.items()}

# ---------------- SELECT QUESTIONS ----------------
def select_questions(df, weights, total_q):
    questions = []
    for domain, weight in weights.items():
        n = max(1, int(weight * total_q))  # normalized weight
        domain_q = df[df["domain"] == domain]
        if not domain_q.empty:
            sampled = domain_q.sample(min(n, len(domain_q)))
            questions.extend(sampled.to_dict(orient="records"))
    if len(questions) < total_q:
        remaining = total_q - len(questions)
        extra = df.sample(min(remaining, len(df))).to_dict(orient="records")
        questions.extend(extra)
    return questions[:total_q]

# ---------------- SIMILARITY ----------------
def get_similarity(a, b):
    tfidf = TfidfVectorizer()
    vec = tfidf.fit_transform([a, b])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

# -------- FILLER WORD COUNT --------
def count_fillers(text):
    fillers = ["um", "uh", "like", "you know", "basically"]
    return sum(text.lower().count(f) for f in fillers)

# -------- TECHNICAL DEPTH --------
def technical_depth(text):
    tech_words = [
        "algorithm", "complexity", "api", "database",
        "model", "training", "optimization", "scalability"
    ]
    return sum(word in text.lower() for word in tech_words)

# -------- KEYWORD MATCH --------
def keyword_match(user, keywords):
    user_tokens = set(re.findall(r"\w+", user.lower()))
    return sum(1 for k in keywords if k.strip().lower() in user_tokens)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["resume"]
    role = request.form.get("role")
    num_q = int(request.form.get("num_q", 5))

    text = extract_text_from_pdf(file)
    skills = extract_skills(text)
    if not skills:
        skills = ["Python"]   # fallback

    role_weights = ROLE_WEIGHTS[role]
    weights = normalize_weights(role_weights, skills)

    df = pd.read_csv("questions_dataset.csv", encoding="utf-8")
    df.rename(columns={
        "Domain": "domain",
        "Question": "question",
        "Answer": "ideal_answer",
        "Keywords": "keywords"
    }, inplace=True)

    questions = select_questions(df, weights, num_q)
    return render_template("questions.html", questions=questions)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    results = []
    domain_scores = {}

    i = 1
    while True:
        user = request.form.get(f"answer{i}")
        if not user:
            break
        domain = request.form.get(f"domain{i}")
        ideal = request.form.get(f"ideal{i}")
        keywords_raw = request.form.get(f"keywords{i}")
        keywords = keywords_raw.split(",") if keywords_raw else []

        # -------- FEATURE EXTRACTION --------
        similarity = get_similarity(user, ideal)
        keyword_score = keyword_match(user, keywords)
        length = len(user.split())
        filler = count_fillers(user)
        depth = technical_depth(user)

        # -------- MODEL PREDICTION --------
        
        features = pd.DataFrame(
            [[similarity, keyword_score, length, filler, depth]],
            columns=[
                "similarity_score",
                "keyword_match_score",
                "answer_length",
                "filler_count",
                "technical_depth_score"
            ]
        )
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = max(proba)

        

        results.append({
            "domain": domain,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "similarity": round(similarity, 2),
            "keywords": keyword_score,
            "length": length,
            "filler": filler,
            "depth": depth
        })
        
        if domain not in domain_scores:
            domain_scores[domain] = []
        domain_scores[domain].append(prediction)

        i += 1
        # ---------------- OVERALL SCORE ----------------

    score_map = {"Good": 1, "Average": 0.5, "Poor": 0}
    total_score = sum(score_map[r["prediction"]] for r in results)
    overall_score = round((total_score / len(results)) * 100, 2)
    
    # ---------------- DOMAIN ANALYSIS ----------------
    analysis = {}
    for domain, preds in domain_scores.items():
        avg = sum(score_map[p] for p in preds) / len(preds)
        if avg > 0.7:
            analysis[domain] = "Strong"
        elif avg > 0.4:
            analysis[domain] = "Average"
        else:
            analysis[domain] = "Weak"

    # ---------------- DOMAIN SUGGESTIONS ----------------
    domain_suggestions = {}
    for domain, preds in domain_scores.items():
        avg = sum(score_map[p] for p in preds) / len(preds)
        if avg < 0.4:
            domain_suggestions[domain] = "Focus on fundamentals and practice more questions."
        elif avg < 0.7:
            domain_suggestions[domain] = "Revise key concepts and improve clarity."
        else:
            domain_suggestions[domain] = "Keep up the good work!"

    # ---------------- DOMAIN CHART DATA ----------------
    domain_chart = {}
    for domain, preds in domain_scores.items():
        avg = sum(score_map[p] for p in preds) / len(preds)
        domain_chart[domain] = round(avg * 100, 2)
    
    # ---------------- GENERATE CHARTS ----------------
    labels = list(domain_chart.keys())
    scores = list(domain_chart.values())

    # Pie chart
    plt.figure(figsize=(6,6))
    plt.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Domain-wise Performance")
    plt.tight_layout()
    plt.savefig("static/domain_pie.png")
    plt.close()

    # Bar chart
    plt.figure(figsize=(8,5))
    plt.bar(labels, scores, color="skyblue", edgecolor="black")
    plt.title("Domain-wise Scores")
    plt.xlabel("Domains")
    plt.ylabel("Score (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("static/domain_bar.png")
    plt.close()

    # ---------------- INSIGHTS ----------------
    avg_similarity = sum(r["similarity"] for r in results) / len(results)
    avg_keywords = sum(r["keywords"] for r in results) / len(results)
    avg_filler = sum(r["filler"] for r in results) / len(results)
    avg_length = sum(r["length"] for r in results) / len(results)

    insights = []
    if avg_similarity < 0.5:
        insights.append("Your answers lack conceptual clarity.")
    if avg_keywords < 2:
        insights.append("Use more technical keywords.")
    if avg_filler > 2:
        insights.append("Reduce filler words (um, like, basically).")
    if avg_length < 8:
        insights.append("Try to elaborate your answers more.")

    # ---------------- RENDER ----------------
    return render_template("result.html",
                           results=results,
                           overall_score=overall_score,
                           analysis=analysis,
                           insights=insights,
                           domain_chart=domain_chart,
                           domain_suggestions=domain_suggestions)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
