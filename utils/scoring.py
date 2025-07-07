from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_score_and_missing_keywords(jd_text, resumes_text, jd_keywords, top_n=5):
    # TF-IDF Embedding
    texts = [jd_text] + resumes_text
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    jd_vec = X[0]
    resume_vecs = X[1:]
    scores = cosine_similarity(jd_vec, resume_vecs)[0]
    results = []
    for i, resume_text in enumerate(resumes_text):
        resume_tokens = set(resume_text.lower().split())
        missing = [kw for kw in jd_keywords if kw.lower() not in resume_tokens]
        results.append({
            'score': float(scores[i]),
            'missingKeywords': missing[:top_n]
        })
    return results
