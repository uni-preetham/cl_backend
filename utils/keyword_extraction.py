import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nlp = spacy.load('en_core_web_sm')

def extract_keywords(text, top_n=20):
    # TF-IDF keywords
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform([text])
    tfidf_keywords = vectorizer.get_feature_names_out()
    # Named entities
    doc = nlp(text)
    entities = set([ent.text for ent in doc.ents if len(ent.text.split()) < 4])
    # Combine and deduplicate
    keywords = list(set(tfidf_keywords).union(entities))
    return keywords
