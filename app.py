from flask import Flask, request, render_template, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pdfplumber
import pickle
import re
import matplotlib.pyplot as plt
import os
from datetime import datetime
from database import get_db_connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    print("Warning: model.pkl not found. Please train the model first.")
    model = None

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
        "DSA": ["dsa", "data structures", "algorithm"],
        "Python": ["python"],
        "Java": ["java"],
        "C++": ["c++", "cpp"],
        "AIML": ["machine learning", "ai", "artificial intelligence"],
        "Data Science": ["statistics", "pandas", "data analysis"],
        "DBMS": ["sql", "database", "dbms", "mysql", "postgresql"],
        "Web Development": ["html", "css", "javascript", "react", "angular"],
        "Backend Development": ["django", "flask", "node", "api", "rest"],
        "Operating Systems": ["os", "process", "thread", "linux"],
        "Computer Networks": ["network", "tcp", "ip", "dns", "http"],
        "System Design": ["scalability", "architecture", "microservices"],
        "Cloud & DevOps": ["aws", "docker", "kubernetes", "jenkins", "ci/cd"],
        "Cybersecurity": ["security", "encryption", "authentication"],
        "OOPS": ["oops", "object oriented", "inheritance"]
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
        n = max(1, int(weight * total_q))
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
    fillers = ["um", "uh", "like", "you know", "basically", "actually"]
    return sum(text.lower().count(f) for f in fillers)

# -------- TECHNICAL DEPTH --------
def technical_depth(text):
    tech_words = [
        "algorithm", "complexity", "api", "database",
        "model", "training", "optimization", "scalability",
        "framework", "architecture", "deployment"
    ]
    return sum(word in text.lower() for word in tech_words)

# -------- KEYWORD MATCH --------
def keyword_match(user, keywords):
    user_tokens = set(re.findall(r"\w+", user.lower()))
    return sum(1 for k in keywords if k.strip().lower() in user_tokens)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return redirect("/login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password)
            )
            conn.commit()
            conn.close()
            return redirect("/login")
        except Exception as e:
            conn.close()
            return f"User already exists or error: {e}"

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["name"]
            return redirect("/dashboard")

        return "Invalid credentials"

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")
    
    # Get username from session or database
    if "username" not in session:
        conn = get_db_connection()
        user = conn.execute(
            "SELECT name FROM users WHERE id = ?", (session["user_id"],)
        ).fetchone()
        conn.close()
        if user:
            session["username"] = user["name"]
        else:
            return redirect("/login")
    
    return render_template("dashboard.html", username=session["username"])

@app.route("/upload", methods=["GET"])
def upload_page():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" not in session:
        return redirect("/login")
    
    if "resume" not in request.files:
        return "No file uploaded"
    
    file = request.files["resume"]
    role = request.form.get("role")
    num_q = int(request.form.get("num_q", 5))

    text = extract_text_from_pdf(file)
    skills = extract_skills(text)
    if not skills:
        skills = ["Python", "DSA"]   # fallback

    role_weights = ROLE_WEIGHTS[role]
    weights = normalize_weights(role_weights, skills)

    df = pd.read_csv("questions_dataset.csv", encoding="utf-8")
    
    # Make sure columns exist
    if "Domain" in df.columns:
        df.rename(columns={
            "Domain": "domain",
            "Question": "question",
            "Answer": "ideal_answer",
            "Keywords": "keywords"
        }, inplace=True)
    
    questions = select_questions(df, weights, num_q)
    return render_template("questions.html", questions=questions, role=role)

@app.route("/progress")
def progress():
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db_connection()
    results = conn.execute(
        "SELECT * FROM results WHERE user_id = ? ORDER BY date",
        (session["user_id"],)
    ).fetchall()
    conn.close()

    scores = [r["score"] for r in results]
    dates = [r["date"] for r in results]

    # Ensure static directory exists
    static_dir = os.path.join(app.root_path, "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    chart_path = os.path.join(static_dir, "progress_line.png")

    # Generate line chart with Matplotlib
    if scores:
        plt.figure(figsize=(8,5))
        plt.plot(dates, scores, marker='o', color='blue', linewidth=2)
        plt.title("Progress Over Time")
        plt.xlabel("Date")
        plt.ylabel("Score (%)")
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
    else:
        # fallback chart if no data
        plt.figure(figsize=(8,5))
        plt.text(0.5, 0.5, "No progress data available", ha="center", va="center", fontsize=16)
        plt.axis("off")
        plt.savefig(chart_path)
        plt.close()

    return render_template(
        "progress.html",
        results=results,
        scores=scores,
        dates=dates,
        chart_filename="progress_line.png"
    )

@app.route("/evaluate", methods=["POST"])
def evaluate():
    if "user_id" not in session:
        return redirect("/login")
    
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
        if model:
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
        else:
            # Fallback if model not available
            if similarity > 0.6 and keyword_score > 2:
                prediction = "Good"
            elif similarity > 0.3 and keyword_score > 1:
                prediction = "Average"
            else:
                prediction = "Poor"
            confidence = 0.7

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
    
    # -------- SAVE RESULT --------
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO results (user_id, score, date) VALUES (?, ?, ?)",
        (session["user_id"], overall_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()
    
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

    static_dir = os.path.join(app.root_path, "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Pie chart
    if scores and sum(scores) > 0:
        plt.figure(figsize=(6,6))
        plt.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("Domain-wise Performance")
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, "domain_pie.png"))
        plt.close()
    else:
        plt.figure(figsize=(6,6))
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.axis("off")
        plt.savefig(os.path.join(static_dir, "domain_pie.png"))
        plt.close()
    
    # Bar chart
    if scores and sum(scores) > 0:
        plt.figure(figsize=(8,5))
        plt.bar(labels, scores, color="skyblue", edgecolor="black")
        plt.title("Domain-wise Scores")
        plt.xlabel("Domains")
        plt.ylabel("Score (%)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, "domain_bar.png"))
        plt.close()
    else:
        plt.figure(figsize=(8,5))
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.axis("off")
        plt.savefig(os.path.join(static_dir, "domain_bar.png"))
        plt.close()

    # ---------------- INSIGHTS ----------------
    avg_similarity = sum(r["similarity"] for r in results) / len(results)
    avg_keywords = sum(r["keywords"] for r in results) / len(results)
    avg_filler = sum(r["filler"] for r in results) / len(results)
    avg_length = sum(r["length"] for r in results) / len(results)

    insights = []
    if avg_similarity < 0.5:
        insights.append("Your answers lack conceptual clarity.")
    else:
        insights.append("There is clarity on concepts. Try to use more technical language.")
    if avg_keywords < 2:
        insights.append("Use more technical keywords in your answers.")
    else:
        insights.append("Good use of technical keywords!")
    if avg_filler > 2:
        insights.append("Reduce filler words (um, like, basically).")
    else:
        insights.append("Good explanation with minimal stammer! Love the confidence!")
    if avg_length < 8:
        insights.append("Try to elaborate your answers more.")
    else:
        insights.append("Well elaborated answers! Good job!")

    # ---------------- RENDER ----------------
    return render_template("result.html",
                           results=results,
                           overall_score=overall_score,
                           analysis=analysis,
                           insights=insights,
                           domain_chart=domain_chart,
                           domain_suggestions=domain_suggestions)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)