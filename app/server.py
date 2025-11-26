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
from flask_cors import CORS


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

def explain_url(url: str, feats: dict, result: str) -> list[str]:
    """
    Build friendly explanations depending on result:
    - If Legitimate: highlight positive / safe indicators.
    - If Phishing: highlight risk factors.
    """
    reasons = []

    # ------------------------------
    # POSITIVE reasons (Legitimate)
    # ------------------------------
    if result == "Legitimate":
        # Uses HTTPS
        if feats.get("has_https"):
            reasons.append("Uses HTTPS (encrypted connection)")

        # Domain looks normal, not IP-based
        if not feats.get("is_ip_host"):
            reasons.append("Uses a normal domain name, not a raw IP address")

        # Domain structure looks typical
        if feats.get("num_subdomains", 0) <= 2 and feats.get("num_dots_host", 0) <= 3:
            reasons.append("Domain structure looks typical (not many nested subdomains)")

        # No phishing keywords
        if feats.get("keyword_hits", 0) == 0:
            reasons.append("No phishing-related keywords found")

        # URL length looks fine
        if feats.get("url_length", 0) <= 80 and feats.get("num_params", 0) <= 3:
            reasons.append("URL length and parameters appear normal")

        # Not a shortener
        if not feats.get("is_shortener"):
            reasons.append("URL does not use a shortener (transparent destination)")

        if not reasons:
            reasons.append("Overall URL structure appears normal")

        return reasons

    # ------------------------------
    # RISK reasons (Phishing)
    # ------------------------------
    u = url.lower()

    suspicious_words = [
        w for w in (
            "login", "verify", "secure", "update", "confirm",
            "account", "bank", "reset", "password"
        ) if w in u
    ]
    if suspicious_words:
        reasons.append(f"Contains suspicious keyword(s): {', '.join(suspicious_words[:3])}")

    if feats.get("is_shortener", 0) == 1:
        reasons.append("Uses a URL shortener (destination hidden)")

    if feats.get("is_ip_host"):
        reasons.append("Uses a raw IP address instead of a domain name")

    if feats.get("num_subdomains", 0) >= 3 or feats.get("num_dots_host", 0) >= 4:
        reasons.append("Many subdomains/dots (can disguise true domain)")

    if feats.get("entropy_url", 0) > 4.0:
        reasons.append("High URL entropy (random-looking string)")

    if feats.get("num_params", 0) > 3 or feats.get("url_length", 0) > 80:
        reasons.append("Long URL or too many query parameters")

    if not reasons:
        reasons.append("Similar to known phishing URL patterns")

    return reasons


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

# Allow Chrome extension (and other origins) to call our API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

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
            reasons = explain_url(url, feats, result)

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

@app.route("/api/scan-urls", methods=["POST"])
def api_scan_urls():
    """
    API endpoint for Chrome extension / other clients.
    Expects JSON: {"urls": ["http://...", "https://..."]}
    Returns JSON with predictions and reasons.
    """
    data = request.get_json(silent=True) or {}
    urls = data.get("urls") or []

    # Clean and deduplicate URLs
    cleaned = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u or len(u) > 2048:
            continue
        if u in seen:
            continue
        seen.add(u)
        cleaned.append(u)

    results = []

    for url in cleaned:
        try:
            feats = extract_features(url)
            x = np.array([to_vector(feats)], dtype=float)
            label, conf = predict_with_confidence(model, x)
            result = "Phishing" if label == 1 else "Legitimate"
            confidence = float(conf)
            reasons = explain_url(url, feats, result)

            # Log to the same SQLite DB as the main web form
            if result in ("Phishing", "Legitimate"):
                log_scan(url, result, confidence)

            results.append({
                "url": url,
                "result": result,
                "confidence": confidence,
                "reasons": reasons,
            })
        except Exception as e:
            results.append({
                "url": url,
                "result": "Error",
                "confidence": 0.0,
                "reasons": [f"Processing error: {e}"],
            })

    return jsonify({"results": results})



# 5) Run the app directly (development mode)
if __name__ == "__main__":
    # debug=True auto-reloads on code changes (dev only)
    app.run(debug=True)
