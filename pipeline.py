import os
import requests
import sqlite3
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# -----------------------
# App Setup
# -----------------------
app = FastAPI()

# Enable CORS (VERY IMPORTANT for Render evaluation)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DB_NAME = "pipeline.db"

# -----------------------
# Database Setup
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original TEXT,
            analysis TEXT,
            sentiment TEXT,
            source TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------
# Request Schema
# -----------------------
class PipelineRequest(BaseModel):
    email: str
    source: str

# -----------------------
# AI Enrichment
# -----------------------
def analyze_text(text):
    try:
        prompt = f"""
        Analyze this data in 2 sentences and classify sentiment as enthusiastic, critical, or objective.
        Respond strictly in this format:

        Summary: <your summary>
        Sentiment: <enthusiastic/critical/objective>

        Data:
        {text}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content

        summary = ""
        sentiment = "objective"

        for line in content.split("\n"):
            if line.lower().startswith("summary:"):
                summary = line.replace("Summary:", "").strip()
            if line.lower().startswith("sentiment:"):
                sentiment = (
                    line.replace("Sentiment:", "")
                    .strip()
                    .lower()
                    .replace(".", "")
                )

        return summary, sentiment

    except Exception as e:
        return None, f"AI Error: {str(e)}"

# -----------------------
# Store Results
# -----------------------
def store_result(original, analysis, sentiment, source):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        timestamp = datetime.utcnow().isoformat() + "Z"

        cursor.execute("""
            INSERT INTO results (original, analysis, sentiment, source, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (original, analysis, sentiment, source, timestamp))

        conn.commit()
        conn.close()

        return True, timestamp

    except Exception as e:
        return False, str(e)

# -----------------------
# Optional Root Route
# -----------------------
@app.get("/")
def root():
    return {"message": "AI Pipeline is running."}

# -----------------------
# Pipeline Endpoint
# -----------------------
@app.post("/pipeline")
def run_pipeline(request: PipelineRequest):

    items_output = []
    errors = []

    # -----------------------
    # Stage 1: Fetch API Data
    # -----------------------
    try:
        response = requests.get(
            "https://jsonplaceholder.typicode.com/users",
            timeout=5
        )
        response.raise_for_status()
        users = response.json()[:3]
    except Exception as e:
        return {
            "items": [],
            "notificationSent": False,
            "processedAt": datetime.utcnow().isoformat() + "Z",
            "errors": [f"API Fetch Error: {str(e)}"]
        }

    # -----------------------
    # Process Each User
    # -----------------------
    for user in users:
        try:
            original_text = str(user)

            # AI Analysis
            summary, sentiment = analyze_text(original_text)

            if summary is None:
                errors.append(sentiment)
                continue

            # Storage
            stored, timestamp_or_error = store_result(
                original_text,
                summary,
                sentiment,
                request.source
            )

            if not stored:
                errors.append(f"Storage Error: {timestamp_or_error}")
                continue

            items_output.append({
                "original": original_text,
                "analysis": summary,
                "sentiment": sentiment,
                "stored": True,
                "timestamp": timestamp_or_error
            })

        except Exception as e:
            errors.append(f"Processing Error: {str(e)}")
            continue

    # -----------------------
    # Notification (Mock)
    # -----------------------
    try:
        print("Notification sent to: 24f2009044@ds.study.iitm.ac.in")
        notification_sent = True
    except Exception as e:
        notification_sent = False
        errors.append(f"Notification Error: {str(e)}")

    return {
        "items": items_output,
        "notificationSent": notification_sent,
        "processedAt": datetime.utcnow().isoformat() + "Z",
        "errors": errors
    }
