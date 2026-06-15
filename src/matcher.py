from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Lazy loader for sentence-transformers
sbert_model = None

def get_sbert_model():
    global sbert_model
    if sbert_model is None:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sbert_model

def match_tfidf(resume_text, job_desc_text):
    """
    Calculates match percentage using TF-IDF + Cosine Similarity.
    Returns a score between 0 and 100.
    """
    if not resume_text or not job_desc_text:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity * 100), 2)
    except Exception as e:
        print(f"Error in TF-IDF matching: {e}")
        return 0.0

def match_semantic(resume_text, job_desc_text):
    """
    Calculates match percentage using SBERT embedding similarity.
    Returns a score between 0 and 100.
    """
    if not resume_text or not job_desc_text:
        return 0.0
    try:
        model = get_sbert_model()
        embeddings = model.encode([resume_text, job_desc_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        similarity = max(0.0, min(1.0, float(similarity)))
        return round(similarity * 100, 2)
    except Exception as e:
        print(f"Error in Semantic matching: {e}")
        return 0.0

def weighted_match_score(resume_text, jd_text, resume_skills, jd_skills, use_semantic=False, skill_weight=0.6, text_weight=0.4):
    """
    Composite weighted match score combining:
      - Skill Coverage  (custom weight)
      - Text Similarity (custom weight)

    Returns a dict with the composite score and its components.
    """
    # --- Component 1: Skill Coverage ---
    if jd_skills:
        matched = [s for s in resume_skills if s in jd_skills]
        skill_coverage = len(matched) / len(jd_skills)
    else:
        skill_coverage = 0.0

    # --- Component 2: Text Similarity ---
    if use_semantic:
        text_score = match_semantic(resume_text, jd_text) / 100.0
    else:
        text_score = match_tfidf(resume_text, jd_text) / 100.0

    # --- Weighted Composite ---
    composite = (skill_weight * skill_coverage + text_weight * text_score) * 100
    composite = round(composite, 2)

    return {
        "composite_score": composite,
        "skill_coverage_pct": round(skill_coverage * 100, 2),
        "text_similarity_pct": round(text_score * 100, 2),
    }

def score_label(score):
    """Returns a human-readable label and color for a given composite score."""
    if score >= 70:
        return "Excellent Match", "#10b981"
    elif score >= 45:
        return "Good Match", "#f59e0b"
    elif score >= 25:
        return "Partial Match", "#f97316"
    else:
        return "Low Match", "#ef4444"
