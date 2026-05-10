# ============================================
# main.py - AI Grievance Analysis FastAPI
# ============================================
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---- CREATE FASTAPI APP ----
app = FastAPI(
    title="AI Citizen Grievance Analysis System",
    description="Predicts department and sentiment for citizen complaints",
    version="1.0.0"
)

# ---- LOAD SAVED MODELS ----
print("Loading models...")
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
sentiment_vectorizer = pickle.load(
    open('sentiment_vectorizer.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
print("All models loaded successfully!")

# ---- TEXT CLEANING FUNCTIONS ----
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
sentiment_stopwords = stop_words - {
    'not', 'very', 'no', 'never',
    'urgent', 'dangerous', 'please',
    'thank', 'immediately'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w)
             for w in words
             if w not in stop_words]
    return ' '.join(words)

def clean_sentiment_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w)
             for w in words
             if w not in sentiment_stopwords]
    return ' '.join(words)

# ---- INPUT/OUTPUT MODELS ----
class ComplaintInput(BaseModel):
    complaint_text: str

class AnalysisOutput(BaseModel):
    complaint: str
    department: str
    dept_confidence: float
    sentiment: str
    sent_confidence: float
    urgency_score: float
    priority: str
    action_needed: str

# ---- API ENDPOINTS ----

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "AI Citizen Grievance System is Running!",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "ready": True
    }

# Main analysis endpoint
@app.post("/analyze", response_model=AnalysisOutput)
def analyze_complaint(complaint: ComplaintInput):

    text = complaint.complaint_text

    # Department prediction
    dept_cleaned = clean_text(text)
    dept_vectorized = vectorizer.transform([dept_cleaned])
    department = rf_model.predict(dept_vectorized)[0]
    dept_conf = round(
        rf_model.predict_proba(dept_vectorized).max() * 100, 1)

    # Sentiment prediction
    sent_cleaned = clean_sentiment_text(text)
    sent_vectorized = sentiment_vectorizer.transform([sent_cleaned])
    sent_num = svm_model.predict(sent_vectorized)[0]
    sentiment = label_encoder.inverse_transform([sent_num])[0]
    sent_conf = round(
        svm_model.predict_proba(sent_vectorized).max() * 100, 1)

    # Urgency score
    base_scores = {
        'Critical': 90,
        'Negative': 60,
        'Neutral': 30,
        'Positive': 10
    }
    urgency_score = round(
        base_scores[sentiment] * (sent_conf / 100), 1)

    # Priority and action
    if urgency_score >= 70:
        priority = "URGENT"
        action = "Dispatch team immediately today"
    elif urgency_score >= 50:
        priority = "HIGH"
        action = "Schedule repair within this week"
    elif urgency_score >= 25:
        priority = "NORMAL"
        action = "Add to regular maintenance queue"
    else:
        priority = "LOW"
        action = "Acknowledge and monitor"

    return AnalysisOutput(
        complaint=text,
        department=department,
        dept_confidence=dept_conf,
        sentiment=sentiment,
        sent_confidence=sent_conf,
        urgency_score=urgency_score,
        priority=priority,
        action_needed=action
    )