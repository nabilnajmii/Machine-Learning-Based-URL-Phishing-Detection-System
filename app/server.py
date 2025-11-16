# app/server.py
# ------------------------------------------------------------
# Minimal Flask app:
# - loads the trained model saved in models/best_model.joblib
# - shows a form to enter a URL
# - extracts lexical features
# - returns "Phishing" / "Legitimate" with a confidence score
# ------------------------------------------------------------

import os
import sys
from flask import Flask, render_template, request
from joblib import load
import numpy as np
from flask import jsonify

# --- Logging (SQLite) ---
import sqlite3, datetime as dt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "logs.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, url TEXT, result TEXT, confidence REAL
        )
        """)
        conn.commit()

def log_scan(url: str, result: str, confidence: float):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO logs (ts, url, result, confidence) VALUES (?,?,?,?)",
            (dt.datetime.utcnow().isoformat(), url, result, confidence),
        )
        conn.commit()

def explain_url(url: str, feats: dict) -> list[str]:
    reasons = []
    u = url.lower()

    # Keywords commonly seen in phishing
    suspicious_words = [w for w in (
        "login", "verify", "secure", "update", "confirm", 
        "account", "bank", "reset", "password"
    ) if w in u]
    if suspicious_words:
        reasons.append(f"Contains suspicious keyword(s): {', '.join(suspicious_words[:3])}")

    # Uses a shortener
    if feats.get("is_shortener", 0) == 1:
        reasons.append("Uses a URL shortener (destination hidden)")

    # Random-looking string
    if feats.get("entropy_url", 0) > 4.0:
        reasons.append("High URL entropy (random-looking string)")

    # Hyphen trick
    if feats.get("has_hyphen_host", 0) == 1:
        reasons.append("Hyphen in host (often used to mimic brands)")

    # Too many subdomains
    if feats.get("num_subdomains", 0) >= 2 or feats.get("num_dots_host", 0) >= 3:
        reasons.append("Many subdomains/dots (can obscure true domain)")

    # IP address in URL
    if feats.get("is_ip_host", 0) == 1:
        reasons.append("IP address used instead of a domain")

    # Suspicious TLD
    if feats.get("has_suspicious_tld", 0) == 1:
        reasons.append("Suspicious top-level domain")

    # Very long path or too many parameters
    if feats.get("path_length", 0) > 30 or feats.get("num_params", 0) >= 3:
        reasons.append("Unusually long path or many query parameters")

    return reasons[:4]  # Limit to top 4 explanations

# Initialize the database once when the app starts
init_db()

# Make sure Python can find your feature extractor in src/
# Adds the project/src folder to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
sys.path.append(SRC_DIR)

from features import extract_features, to_vector  # noqa

# 1) Create the Flask app
app = Flask(__name__)

# 2) Load the trained model (created by src/train.py)
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
model = load(MODEL_PATH)

# Helper to get probability if the model supports it (SVM with probability=True does)
def predict_with_confidence(model, x_vec: np.ndarray):
    """
    Returns (label, confidence) where label is 0=Legit, 1=Phishing
    Confidence is a number between 0 and 1.
    """
    # Try predict_proba if available
    proba_fn = getattr(getattr(model, "named_steps", {}), "get", lambda *_: None)("clf")
    if hasattr(model, "predict_proba"):
        # pipeline without named_steps['clf'] (e.g., RandomForest only)
        probs = model.predict_proba(x_vec)[0]
        p_phish = float(probs[1])
    elif proba_fn is not None and hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_vec)[0]
        p_phish = float(probs[1])
    else:
        # Fallback: some classifiers may not have predict_proba
        pred = int(model.predict(x_vec)[0])
        p_phish = 0.5 if pred == 1 else 0.5  # neutral fallback
    label = int(p_phish >= 0.5)
    confidence = p_phish if label == 1 else (1.0 - p_phish)
    return label, confidence

# 3) Define the homepage route
@app.route("/", methods=["GET", "POST"])
def index():
    # ⭐ safe defaults so GET doesn't crash
    url = ""
    result = None
    confidence = None
    reasons = []   # ⭐ define reasons for both GET and POST

    if request.method == "POST":
        url = (request.form.get("url") or "").strip()
        if len(url) > 2048:
            result = "Input too long"
            confidence = 0.0
        else:
            # feature extraction + prediction
            feats = extract_features(url)
            x = np.array([to_vector(feats)], dtype=float)
            label, conf = predict_with_confidence(model, x)
            result = "Phishing" if label == 1 else "Legitimate"
            confidence = conf

            # ⭐ build explainability snippets
            reasons = explain_url(url, feats)

            # log only valid predictions
            if result in ("Phishing", "Legitimate") and confidence is not None:
                log_scan(url, result, float(confidence))

    # ⭐ always pass reasons (list) to the template
    return render_template("index.html", url=url, result=result,
                           confidence=confidence, reasons=reasons)


@app.route("/history")
def history():
    rows = []
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for r in c.execute("SELECT ts, url, result, confidence FROM logs ORDER BY id DESC LIMIT 100"):
            rows.append({"ts": r[0], "url": r[1], "result": r[2], "confidence": r[3]})
    return render_template("history.html", rows=rows)

@app.route("/dashboard")
def dashboard():
    stats = {"phishing": 0, "legit": 0, "total": 0}

    # read aggregate counts from logs.db
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for r in c.execute("SELECT result, COUNT(*) FROM logs GROUP BY result"):
            if r[0] == "Phishing":
                stats["phishing"] = r[1]
            elif r[0] == "Legitimate":
                stats["legit"] = r[1]
    stats["total"] = stats["phishing"] + stats["legit"]

    return render_template("dashboard.html", stats=stats)


# 5) Run the app directly (development mode)
if __name__ == "__main__":
    # debug=True auto-reloads on code changes (dev only)
    app.run(debug=True)
