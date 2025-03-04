import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------
# 1) Scraper
# -----------
def scrape_website(url):
    """
    Scrapes the given URL and returns a dict with:
      - 'title': <title> text
      - 'description': <meta name='description'> content
      - 'snippet': up to 1000 chars of visible text
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises HTTPError if status is 4xx/5xx

        soup = BeautifulSoup(response.text, "html.parser")

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else "No title found"

        # Description
        desc_tag = soup.find("meta", attrs={"name": "description"})
        description = desc_tag["content"].strip() if (desc_tag and "content" in desc_tag.attrs) else "No description found"

        # Snippet from page text
        full_text = soup.get_text(separator=' ', strip=True)
        snippet = full_text[:1000]  # first 1000 chars

        return {"title": title, "description": description, "snippet": snippet}

    except Exception as e:
        print(f"Failed to scrape {url}. Reason: {e}")
        return None

# -----------
# 2) Text Cleaner
# -----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text, re.I|re.A)
    text = re.sub(r'\\s+', ' ', text)
    return text.strip()

# -----------
# 3) Load Model + Vectorizer
# -----------
# Adjust paths to point to wherever your .pkl files are
MODEL_PATH = '/Users/bryce/Desktop/INLS697/INLS697_proj/models/logistic_regression_model_2_labels.pkl'
VECTORIZER_PATH = '/Users/bryce/Desktop/INLS697/INLS697_proj/models/tfidf_vectorizer_2_labels.pkl'
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -----------
# 4) Analysis Function
# -----------
def analyze_bias(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred_class = model.predict(vectorized)[0]
    pred_proba = model.predict_proba(vectorized)[0]
    # For probability output, label -> probability
    class_labels = model.classes_
    probabilities = dict(zip(class_labels, pred_proba))
    return pred_class, probabilities

# -----------
# 5) Streamlit App
# -----------
def main():
    st.title("Political Bias Analyzer")

    url = st.text_input("Enter a URL to analyze")

    if st.button("Analyze"):
        # 1) Scrape
        scraped_data = scrape_website(url)
        if not scraped_data:
            st.error("Failed to scrape the URL. Check the link and try again.")
            return

        # 2) Combine text for analysis
        combined_text = " ".join([scraped_data["title"], scraped_data["description"], scraped_data["snippet"]])
        
        # 3) Analyze
        pred_label, probabilities = analyze_bias(combined_text)

        # 4) Display result
        st.write(f"**Predicted Bias:** {pred_label}")
        st.write("**Probabilities:**")
        for label, prob in probabilities.items():
            st.write(f"- {label}: {prob:.4f}")

if __name__ == "__main__":
    main()
