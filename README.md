# infotact-project1-grievance-nlp
AI-Powered Citizen Grievance &amp; Sentiment Analysis System | NLP Project | Infotact Internship
# AI Citizen Grievance & Sentiment Analysis System
## Infotact Technical Internship - Project 1

## What this project does
An AI-powered system that automatically analyzes 
citizen complaints and:
- Predicts which government department should handle it
- Analyzes sentiment (Critical/Negative/Neutral/Positive)
- Assigns urgency score (0-100)
- Recommends priority action

## Tech Stack
- Python, NLTK, Scikit-learn
- TF-IDF Vectorization
- Random Forest (Department Classifier)
- SVM (Sentiment Analyzer)
- FastAPI (REST API)

## Models Performance
- Department Classifier: 53.33% (Random Forest)
- Sentiment Analyzer: 66.11% (SVM wins)
- Macro F1 Score: 53.33%

## API Endpoints
- GET  /          → System status
- GET  /health    → Health check
- POST /analyze   → Analyze complaint

## How to run
pip install -r requirements.txt
uvicorn main:app --reload

## Week-wise Progress
- Week 1: Data collection, EDA, Word Cloud
- Week 2: TF-IDF, Department classifier
- Week 3: Sentiment analysis, Urgency scoring
- Week 4: FastAPI deployment, Final evaluation

## GitHub Commits
All 4 weeks of daily commits maintained ✅