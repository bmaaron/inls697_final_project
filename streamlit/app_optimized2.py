import streamlit as st
# Must set page config as the first Streamlit command
st.set_page_config(page_title="Political Leaning Classifier", page_icon="📰", layout="wide")

import torch
import requests
from bs4 import BeautifulSoup
import re
import html
import numpy as np
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from collections import Counter

# Load NLTK resources more carefully
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    
    # Check if nltk data exists and download if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Downloading NLTK tokenizer...")
        nltk.download('punkt', quiet=True)
    
    # Check for stopwords
    try:
        from nltk.corpus import stopwords
        stopwords.words('english')
    except (LookupError, ImportError):
        st.info("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
        # Reload after download
        from nltk.corpus import stopwords
except Exception as e:
    st.error(f"Error loading NLTK: {str(e)}")
    stopwords = None

# Define labels
LABELS = ["Left", "Center", "Right"]

# Map topics to typical political associations
# This could be derived from training data or defined manually
TOPIC_POLITICAL_ASSOCIATIONS = {
    # Religious topics often associated with right
    'christian': 2, 'christianity': 2, 'religion': 2, 'faith': 2, 'church': 2, 'bible': 2, 'gospel': 2,
    'jesus': 2, 'god': 2, 'prayer': 2, 'easter': 2, 'christmas': 2,
    # Economic topics with various associations
    'tax': 2, 'taxes': 2, 'regulation': 0, 'welfare': 0, 'government': 1, 
    'economy': 1, 'economic': 1, 'budget': 1, 'deficit': 2, 'spending': 1,
    # Social topics 
    'equality': 0, 'justice': 0, 'rights': 0, 'abortion': 0, 'lgbt': 0, 
    'transgender': 0, 'progressive': 0, 'diversity': 0, 'inclusion': 0,
    # Immigration topics
    'immigration': 2, 'border': 2, 'immigrant': 2, 'migrants': 1,
    # Environmental topics
    'climate': 0, 'environment': 0, 'fossil': 0, 'renewable': 0, 'green': 0,
    # Gun topics
    'gun': 2, 'guns': 2, 'second amendment': 2, 'nra': 2,
    # Healthcare
    'healthcare': 0, 'universal': 0, 'medicare': 0, 'insurance': 1,
    # Foreign policy
    'military': 2, 'defense': 2, 'terrorism': 1, 'foreign': 1,
}

# Lists for stance detection
POSITIVE_MARKERS = [
    'support', 'agree', 'positive', 'good', 'favor', 'benefit', 'advocate', 'defend', 
    'praise', 'celebrate', 'champion', 'uphold', 'embrace', 'promote', 'endorse'
]

NEGATIVE_MARKERS = [
    'criticize', 'against', 'reject', 'oppose', 'negative', 'bad', 'condemn', 'problem',
    'denounce', 'attack', 'protest', 'disapprove', 'dispute', 'dismiss', 'mock',
    'ridicule', 'undermine', 'challenge', 'question', 'controversial', 'fail', 'wrong'
]

@st.cache_resource
def load_model(model_path=""):
    """Load model and tokenizer, cached with streamlit"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # If no path specified, use a demo approach
    if not model_path:
        st.warning("No model path specified. Using demo mode with simulated predictions.")
        # Create a dummy model and tokenizer for demo purposes
        class DummyModel:
            def __call__(self, **kwargs):
                # Create a dummy output with random logits
                class DummyOutput:
                    def __init__(self):
                        self.logits = torch.tensor([[np.random.normal(0, 1) for _ in range(3)]])
                return DummyOutput()
        
        class DummyTokenizer:
            def __call__(self, text, **kwargs):
                # Create a dummy encoding
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5] * 10])}
            
            def tokenize(self, text):
                # Create dummy tokens
                return text.split()
            
            def decode(self, ids):
                # Create dummy decode
                return "Decoded text"
        
        return DummyModel(), DummyTokenizer()
    
    # Load the actual model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise

# --- Topic and Stance Analysis Functions ---
def analyze_topic_and_stance(text):
    """Analyze both the topics and the stance taken toward those topics."""
    if not text:
        return []
    
    # 1. Extract main topics using keyword frequency
    words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
    try:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    except:
        # Fallback if stopwords aren't available
        common_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                       'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                       'this', 'that', 'these', 'those', 'it', 'they', 'them', 'their', 'from'}
        words = [w for w in words if w not in common_words]
    
    potential_topics = Counter(words).most_common(15)
    
    # 2. For each potential topic, analyze stance markers around it
    topic_stance_pairs = []
    
    for topic, count in potential_topics:
        # Only consider topics that appear frequently enough
        if count < 2:
            continue
            
        # Skip very common but non-informative words
        if topic in ['said', 'says', 'also', 'many', 'people', 'would', 'could', 'article']:
            continue
            
        # Look for stance markers before or after topic mentions
        mentions = re.finditer(r'\b' + topic + r'\b', text.lower())
        stance_scores = []
        
        for match in mentions:
            # Get context window (60 chars before and after)
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            context = text[start:end]
            
            # Check for positive/negative stance markers
            p_score = sum(1 for marker in POSITIVE_MARKERS if marker in context.lower())
            n_score = sum(1 for marker in NEGATIVE_MARKERS if marker in context.lower())
            
            # Calculate net stance (-1 to 1)
            if p_score > 0 or n_score > 0:
                stance = (p_score - n_score) / (p_score + n_score)
                stance_scores.append(stance)
        
        # Average the stance if we found any
        if stance_scores:
            avg_stance = sum(stance_scores) / len(stance_scores)
            topic_stance_pairs.append((topic, avg_stance))
    
    return topic_stance_pairs

def stance_aware_classification(pred_idx, probs, text, topic_stance_pairs):
    """Adjust classification based on stance detection results."""
    
    # Calculate total stance-based adjustment
    adjustment = 0.0
    explanation = []
    
    for topic, stance in topic_stance_pairs:
        # Find closest matching topic in our associations dictionary
        matching_topic = None
        for known_topic in TOPIC_POLITICAL_ASSOCIATIONS:
            if known_topic in topic or topic in known_topic:
                matching_topic = known_topic
                break
                
        if matching_topic:
            associated_leaning = TOPIC_POLITICAL_ASSOCIATIONS[matching_topic]
            
            # If stance is opposite of the typical association, apply an adjustment
            # Stance: -1 (against) to 1 (supportive)
            if stance < -0.2 and associated_leaning == 2:  # Critical of right topic
                # Push toward left
                adjustment -= 0.15
                explanation.append(f"Critical stance on '{topic}' (typically {LABELS[associated_leaning]})")
            elif stance < -0.2 and associated_leaning == 1:  # Critical of center topic
                # Push slightly toward edges
                adjustment += -0.05
                explanation.append(f"Critical stance on '{topic}' (typically {LABELS[associated_leaning]})")
            elif stance > 0.2 and associated_leaning == 2:  # Supportive of right topic
                # Push toward right
                adjustment += 0.15
                explanation.append(f"Supportive stance on '{topic}' (typically {LABELS[associated_leaning]})")
            elif stance > 0.2 and associated_leaning == 0:  # Supportive of left topic
                # Push toward left
                adjustment -= 0.15
                explanation.append(f"Supportive stance on '{topic}' (typically {LABELS[associated_leaning]})")
    
    # Apply the adjustment to probabilities
    adjusted_probs = probs.clone()
    
    # Cap the adjustment to avoid extreme shifts
    adjustment = max(min(adjustment, 0.3), -0.3)
    
    if adjustment > 0:  # Push toward right
        shift = min(adjusted_probs[0] * 0.5, adjustment)  # Take from left
        adjusted_probs[0] -= shift
        adjusted_probs[2] += shift
    elif adjustment < 0:  # Push toward left
        shift = min(adjusted_probs[2] * 0.5, -adjustment)  # Take from right
        adjusted_probs[2] -= shift
        adjusted_probs[0] += shift
    
    # Renormalize
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    # Recalculate prediction
    new_pred_idx = torch.argmax(adjusted_probs).item()
    
    return new_pred_idx, adjusted_probs, explanation

# --- Optimized Scraping ---
def extract_article(url, method="auto"):
    """Extract article using specified method with fallbacks"""
    if method == "auto" or method == "newspaper3k":
        try:
            # Only import if needed
            import newspaper
            article = newspaper.Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 200:
                return {
                    "title": article.title,
                    "text": article.text,
                    "method": "newspaper3k"
                }
        except Exception as e:
            if method != "auto":  # Only show warning if specifically requested
                st.warning(f"newspaper3k extraction failed: {e}")
    
    if method == "auto" or method == "trafilatura":
        try:
            # Only import if needed
            import trafilatura
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                # Try text extraction first (faster)
                result = trafilatura.extract(downloaded, output_format='text')
                if result and len(result) > 200:
                    return {
                        "title": "",  # No title in text-only mode
                        "text": result,
                        "method": "trafilatura-text"
                    }
        except Exception as e:
            if method != "auto":  # Only show warning if specifically requested
                st.warning(f"trafilatura extraction failed: {e}")
    
    if method == "auto" or method == "beautifulsoup":
        try:
            response = requests.get(url, timeout=5, 
                                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Get title
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            # Try to find main content - simplified approach
            main_content = soup.find('article') or soup.find('main') or soup.find('body')
            
            if main_content:
                # Remove obvious noise
                for noise in main_content.select('script, style, nav, footer'):
                    noise.decompose()
                    
                # Get text from paragraphs
                paragraphs = [p.get_text(strip=True) for p in main_content.find_all('p') if len(p.get_text(strip=True)) > 50]
                text = ' '.join(paragraphs)
                
                # Basic cleanup
                text = html.unescape(text)
                text = re.sub(r'\s+', ' ', text)
                
                if len(text) > 200:
                    return {
                        "title": title,
                        "text": text,
                        "method": "beautifulsoup"
                    }
            
        except Exception as e:
            if method != "auto":  # Only show warning if specifically requested
                st.warning(f"BeautifulSoup extraction failed: {e}")
    
    # If we get here, all methods failed
    st.error(f"Failed to extract content from {url}")
    return None

def clean_article_text(text):
    """Clean article text with minimal processing"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common boilerplate phrases (minimal set)
    for phrase in ["subscribe to our newsletter", "sign up for our", "follow us on", "share this article"]:
        text = re.sub(f"{phrase}.*?\\.", "", text, flags=re.IGNORECASE)
    
    return text

# --- Optimized Chunking ---
def create_chunks(text, _tokenizer, max_tokens=512):
    """Create chunks based on paragraphs and sentences"""
    # Handle the case where tokenizer might be a dummy object
    if not hasattr(_tokenizer, 'tokenize'):
        # Create a simple dummy function
        def simple_tokenize(text):
            return text.split()
        _tokenizer.tokenize = simple_tokenize
    
    # Split into paragraphs
    paragraphs = re.split(r'\n+', text)
    chunks = []
    chunk_positions = []  # beginning, middle, end
    
    # Process paragraphs
    total_paragraphs = len([p for p in paragraphs if p.strip()])
    current_para_idx = 0
    
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        current_para_idx += 1
        para_position = "beginning" if current_para_idx < total_paragraphs * 0.25 else \
                        "end" if current_para_idx > total_paragraphs * 0.75 else "middle"
        
        # Try to tokenize safely
        try:
            para_tokens = _tokenizer.tokenize(para)
            para_len = len(para_tokens)
        except Exception as e:
            # Fallback if tokenization fails
            para_len = len(para.split())
        
        # If paragraph fits in current chunk
        if current_len + para_len <= max_tokens:
            current_chunk.append(para)
            current_len += para_len
        else:
            # Add current chunk if not empty
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                chunk_positions.append(para_position)
                current_chunk = []
                current_len = 0
            
            # Handle paragraphs that are too long
            if para_len > max_tokens:
                # Split into sentences
                try:
                    sentences = sent_tokenize(para)
                except:
                    # Fallback if sentence tokenization fails
                    sentences = re.split(r'[.!?]+', para)
                
                current_sent_chunk = []
                current_sent_len = 0
                
                for sent in sentences:
                    if not sent.strip():
                        continue
                        
                    # Try to tokenize safely
                    try:
                        sent_tokens = _tokenizer.tokenize(sent)
                        sent_len = len(sent_tokens)
                    except Exception as e:
                        # Fallback if tokenization fails
                        sent_len = len(sent.split())
                    
                    if current_sent_len + sent_len <= max_tokens:
                        current_sent_chunk.append(sent)
                        current_sent_len += sent_len
                    else:
                        if current_sent_chunk:
                            chunks.append(" ".join(current_sent_chunk))
                            chunk_positions.append(para_position)
                            current_sent_chunk = []
                            current_sent_len = 0
                        
                        # Handle very long sentences with simple truncation
                        if sent_len > max_tokens:
                            # Simple truncation if decoder not available
                            if hasattr(_tokenizer, 'decode'):
                                try:
                                    sent_trunc = _tokenizer.decode(
                                        _tokenizer(sent, truncation=True, max_length=max_tokens)["input_ids"]
                                    )
                                except:
                                    sent_trunc = sent[:200] + "..."
                            else:
                                sent_trunc = sent[:200] + "..."
                                
                            chunks.append(sent_trunc)
                            chunk_positions.append(para_position)
                        else:
                            current_sent_chunk.append(sent)
                            current_sent_len = sent_len
                
                if current_sent_chunk:
                    chunks.append(" ".join(current_sent_chunk))
                    chunk_positions.append(para_position)
            else:
                # Start new chunk with this paragraph
                current_chunk.append(para)
                current_len = para_len
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        chunk_positions.append("end" if "end" not in chunk_positions else "middle")
    
    return chunks, chunk_positions

# --- Enhanced Classification with Stance Analysis ---
def classify_weighted_chunks(chunks, positions, _tokenizer, model):
    """Classify chunks with position-based weighting and stance analysis"""
    # Type-based weighting factors (simplified)
    position_weights = {
        "beginning": 1.5,  # Higher weight for beginning
        "middle": 1.0,     # Base weight
        "end": 1.3         # Higher weight for conclusion
    }
    
    # Handle special positions
    if chunks and len(chunks) > 0:
        # Give title/first chunk more weight
        position_weights["beginning"] = 2.0
    
    # Progress bar for chunk processing
    progress_text = "Analyzing article chunks..."
    chunk_progress = st.progress(0.0, text=progress_text)
    
    logits_list = []
    weights = []
    chunk_details = []
    
    for i, (chunk, position) in enumerate(zip(chunks, positions)):
        # Skip empty chunks
        if not chunk.strip():
            continue
        
        # Update progress
        chunk_progress.progress((i + 1) / len(chunks), text=f"{progress_text} ({i+1}/{len(chunks)})")
        
        try:
            # Encode and predict
            encodings = _tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                output = model(**encodings)
            
            logits = output.logits.squeeze().numpy()
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
            pred_idx = torch.argmax(probs).item()
        except Exception as e:
            # If model prediction fails, use random prediction as fallback
            st.warning(f"Error in model prediction, using fallback: {str(e)}")
            logits = np.random.normal(0, 1, size=3)
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
            pred_idx = torch.argmax(probs).item()
        
        # Simple weighting based on position
        weight = position_weights.get(position, 1.0)
        
        # Store results
        logits_list.append(logits)
        weights.append(weight)
        
        # Store details for visualization
        chunk_details.append({
            "index": i + 1,
            "position": position,
            "weight": weight,
            "length": len(encodings["input_ids"][0]) if isinstance(encodings, dict) and "input_ids" in encodings else 0,
            "confidence": probs[pred_idx].item(),
            "prediction": LABELS[pred_idx],
            "text": chunk[:100] + "..." if len(chunk) > 100 else chunk
        })
    
    # Clear progress bar
    chunk_progress.empty()
    
    # Weighted average of logits
    if logits_list:
        weighted_logits = np.average(logits_list, axis=0, weights=weights)
        initial_probs = torch.nn.functional.softmax(torch.tensor(weighted_logits), dim=0)
        initial_pred_idx = torch.argmax(initial_probs).item()
    else:
        # Fallback for empty result
        initial_probs = torch.tensor([0.33, 0.34, 0.33])  # Center bias as fallback
        initial_pred_idx = 1  # Center
    
    # Analyze topic and stance across all chunks
    full_text = " ".join(chunks) if chunks else ""
    topic_stance_pairs = analyze_topic_and_stance(full_text)
    
    # Apply stance-aware adjustments
    final_pred_idx, final_probs, explanations = stance_aware_classification(
        initial_pred_idx, initial_probs, full_text, topic_stance_pairs
    )
    
    return final_pred_idx, final_probs, chunk_details, topic_stance_pairs, explanations, initial_pred_idx, initial_probs

# --- Optional Word Attribution ---
def generate_word_attribution(text, explainer, max_length=500):
    """Generate word attribution for a sample of text with white background container"""
    try:
        # Use a larger portion of the text
        sample_text = text[:2500]  # Use more of the beginning of the text
        truncated_sample = sample_text[:max_length]
        
        # Generate attribution
        word_attributions = explainer(truncated_sample)
        viz = explainer.visualize()
        
        # Check if visualization is HTML or matplotlib
        if hasattr(viz, 'savefig'):
            return {"attributions": word_attributions, "viz": viz, "type": "matplotlib"}
        else:
            # Create a white box container for the HTML visualization
            if hasattr(viz, '_repr_html_'):
                original_html = viz._repr_html_()
                
                # Create a wrapper with explicit white background and dark text
                white_box_html = f"""
                <div style="background-color: white; color: black; padding: 20px; border-radius: 5px; margin: 10px 0;">
                    <style>
                        /* Force all text to be dark */
                        .word-attributions-wrapper * {{
                            color: black !important;
                        }}
                        /* Better contrast for positive/negative highlighting */
                        .positive {{
                            background-color: rgba(0, 200, 0, 0.4) !important;
                            color: black !important;
                        }}
                        .negative {{
                            background-color: rgba(255, 0, 0, 0.4) !important;
                            color: black !important;
                        }}
                        .neutral {{
                            background-color: rgba(200, 200, 200, 0.4) !important;
                            color: black !important;
                        }}
                        /* Override any table styling */
                        table.attributions tr td, 
                        table.attributions tr th {{
                            color: black !important;
                            border: 1px solid #ddd !important;
                            padding: 8px !important;
                        }}
                        table.attributions {{
                            border-collapse: collapse !important;
                            width: 100% !important;
                            margin-bottom: 15px !important;
                        }}
                        table.attributions th {{
                            background-color: #f2f2f2 !important;
                            font-weight: bold !important;
                        }}
                    </style>
                    {original_html}
                </div>
                """
                
                # Create a custom HTML object with our white box wrapper
                from IPython.core.display import HTML
                custom_html = HTML(white_box_html)
                
                return {"attributions": word_attributions, "viz": custom_html, "type": "html"}
            else:
                return {"attributions": word_attributions, "viz": viz, "type": "html"}
            
    except Exception as e:
        st.warning(f"Attribution generation failed: {str(e)}")
        return None

# --- Visualization Functions ---
def plot_chunk_predictions(chunk_details):
    """Create visualization of chunk predictions"""
    if not chunk_details:
        return None
        
    df = pd.DataFrame(chunk_details)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create a categorical color map based on predictions
    cmap = {'Left': '#3182bd', 'Center': '#bdbdbd', 'Right': '#e6550d'}
    df['color'] = df['prediction'].map(cmap)
    
    # Create scatter plot
    ax.scatter(df.index, df['confidence'], 
              s=df['weight']*100,  # Size based on weight
              c=df['color'],       # Color based on prediction
              alpha=0.7)
    
    # Add chunk position labels
    for i, row in df.iterrows():
        ax.annotate(row['position'], (i, row['confidence']), 
                   fontsize=8, ha='center', va='bottom')
    
    plt.title('Chunk Analysis')
    plt.xlabel('Chunk Index')
    plt.ylabel('Confidence')
    plt.ylim(0, 1.05)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=8)
        for label, color in cmap.items()
    ]
    plt.legend(handles=legend_elements, title='Prediction')
    
    return fig

def plot_topic_stance(topic_stance_pairs):
    """Create visualization of topic stances"""
    if not topic_stance_pairs:
        return None
    
    # Sort by stance for better visualization
    topic_stance_pairs.sort(key=lambda x: x[1])
    
    topics = [pair[0] for pair in topic_stance_pairs]
    stances = [pair[1] for pair in topic_stance_pairs]
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(topics) * 0.3)))
    
    # Create horizontal bar chart
    bars = ax.barh(topics, stances, height=0.7)
    
    # Color the bars based on stance
    for i, bar in enumerate(bars):
        stance = stances[i]
        if stance > 0.2:
            bar.set_color('#4CAF50')  # Green for positive
        elif stance < -0.2:
            bar.set_color('#F44336')  # Red for negative
        else:
            bar.set_color('#9E9E9E')  # Grey for neutral
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.xlim(-1, 1)
    plt.title('Topic-Stance Analysis')
    plt.xlabel('Stance (Negative → Positive)')
    
    return fig

def process_text_input(user_input, model, tokenizer, explainer, generate_attribution, viz_type):
    """Process text input for classification"""
    start_time = time.time()
    
    # Clean and chunk the text
    cleaned_text = clean_article_text(user_input)
    chunks, positions = create_chunks(cleaned_text, tokenizer)
    
    # Classify chunks with stance analysis
    pred_idx, probs, chunk_details, topic_stance_pairs, explanations, initial_pred_idx, initial_probs = classify_weighted_chunks(
        chunks, positions, tokenizer, model
    )
    
    # Optionally generate attribution
    attribution_result = None
    if generate_attribution and explainer and chunks:
        with st.spinner("Generating word attribution..."):
            # Use a larger portion of the entire text for attribution
            # Using the whole text instead of just the first chunk
            full_text = " ".join(chunks)
            attribution_result = generate_word_attribution(
                full_text, explainer, max_length=500
            )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Display results
    display_results(
        pred_idx, probs, chunk_details, 
        attribution_result, viz_type, 
        user_input, processing_time,
        topic_stance_pairs, explanations,
        initial_pred_idx, initial_probs
    )

def process_url_input(url, method_param, model, tokenizer, explainer, generate_attribution, viz_type):
    """Process URL input for classification"""
    start_time = time.time()
    
    # Extract content based on selected method
    data = extract_article(url, method=method_param)
    
    if not data:
        st.error("Failed to extract content from the URL.")
        return
    
    # Show extraction method
    st.info(f"Content extracted using: {data.get('method', 'unknown')}")
    
    # Clean and chunk the text
    cleaned_text = clean_article_text(data.get("text", ""))
    chunks, positions = create_chunks(cleaned_text, tokenizer)
    
    # Classify chunks with stance analysis
    pred_idx, probs, chunk_details, topic_stance_pairs, explanations, initial_pred_idx, initial_probs = classify_weighted_chunks(
        chunks, positions, tokenizer, model
    )
    
    # Optionally generate attribution
    attribution_result = None
    if generate_attribution and explainer and chunks:
        with st.spinner("Generating word attribution..."):
            # Use a larger portion of the entire text for attribution
            full_text = " ".join(chunks)
            attribution_result = generate_word_attribution(
                full_text, explainer, max_length=500
            )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Display results
    display_results(
        pred_idx, probs, chunk_details, 
        attribution_result, viz_type, 
        data, processing_time,
        topic_stance_pairs, explanations,
        initial_pred_idx, initial_probs
    )
def display_results(pred_idx, probs, chunk_details, attribution_result, viz_type, 
                   input_data, processing_time, topic_stance_pairs, explanations,
                   initial_pred_idx, initial_probs):
    """Display classification results with added topic-stance analysis"""
    
    # Show processing time
    st.info(f"⏱️ Processing completed in {processing_time:.1f} seconds")
    
    # Create columns for results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Main prediction
        st.success(f"## Predicted leaning: **{LABELS[pred_idx]}**")
        
        # Show if prediction was adjusted
        if pred_idx != initial_pred_idx:
            st.warning(f"Initial prediction (**{LABELS[initial_pred_idx]}**) was adjusted based on topic-stance analysis")
        
        # Confidence scores as bar chart
        st.write("### Confidence scores:")
        chart_data = pd.DataFrame({
            "Leaning": LABELS,
            "Score": probs.numpy()
        })
        chart_data = chart_data.set_index("Leaning")
        st.bar_chart(chart_data)
        
        # Display attribution if available
        if attribution_result:
            st.write("### Word Attribution Analysis")
            
            # Handle different visualization types
            if attribution_result["type"] == "matplotlib":
                st.pyplot(attribution_result["viz"])
            elif attribution_result["type"] == "html":
                # Use components.v1.html for HTML visualizations
                # Increase height to show more of the attribution
                st.components.v1.html(attribution_result["viz"]._repr_html_(), height=600)
                
                # Add explanation of the color coding
                st.info("""
                **Word Attribution Legend:**
                - 🟢 **Green** highlights indicate words that push the prediction toward the currently predicted class
                - 🔴 **Red** highlights indicate words that push away from the predicted class
                - ⚪ **Gray/neutral** words have minimal impact on the classification
                
                This visualization helps identify which specific words and phrases are influencing the model's political leaning classification.
                """)
    
    with col2:
        # Topic and Stance Analysis
        if topic_stance_pairs:
            st.write("### Topic-Stance Analysis")
            
            # Create topic-stance visualization
            topic_viz = plot_topic_stance(topic_stance_pairs)
            if topic_viz:
                st.pyplot(topic_viz)
                
                # Add explanation about the chart
                st.info("""
                **Topic-Stance Chart:**
                - 🟢 **Green bars** indicate topics the article is supportive of
                - 🔴 **Red bars** indicate topics the article is critical of
                - ⚪ **Gray bars** indicate neutral stance
                
                When a topic is typically associated with one political leaning, but the article takes an opposite stance toward it, this may affect classification.
                """)
            
            # Show adjustment explanations
            if explanations:
                st.write("#### Classification Adjustments")
                for explanation in explanations:
                    st.info(explanation)
        
        # Chunk visualization
        st.write("### Chunk Analysis")
        
        # Create chunk visualization
        chunk_viz = plot_chunk_predictions(chunk_details)
        if chunk_viz:
            st.pyplot(chunk_viz)
        
        # Summary statistics
        if chunk_details:
            st.write("### Chunk Breakdown")
            
            # Count predictions by type
            pred_counts = {}
            for detail in chunk_details:
                pred = detail["prediction"]
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
                
            # Create summary statistics
            pred_df = pd.DataFrame({
                "Leaning": list(pred_counts.keys()),
                "Count": list(pred_counts.values())
            })
            
            # Display as a small table
            st.table(pred_df)
    
    # Detailed view
    if viz_type == "Detailed":
        st.write("## Detailed Analysis")
        
        # Topic-Stance details
        if topic_stance_pairs:
            st.write("### Topic-Stance Details")
            
            # Create a DataFrame for better visualization
            stance_df = pd.DataFrame(
                [(topic, f"{stance:.2f}", "Supportive" if stance > 0.2 else "Critical" if stance < -0.2 else "Neutral") 
                 for topic, stance in topic_stance_pairs],
                columns=["Topic", "Score", "Stance"]
            )
            
            st.dataframe(stance_df, use_container_width=True)
            
            # Show conflicting signals warning
            conflicts = []
            for topic, stance in topic_stance_pairs:
                # Find closest matching topic in our associations dictionary
                for known_topic in TOPIC_POLITICAL_ASSOCIATIONS:
                    if known_topic in topic or topic in known_topic:
                        associated_leaning = TOPIC_POLITICAL_ASSOCIATIONS[known_topic]
                        # Check for conflict
                        if stance < -0.2 and associated_leaning == 2:  # Critical of right topic
                            conflicts.append(f"Critical stance on '{topic}' (typically {LABELS[associated_leaning]})")
                        elif stance > 0.2 and associated_leaning == 0:  # Supportive of left topic
                            conflicts.append(f"Supportive stance on '{topic}' (typically {LABELS[associated_leaning]})")
                        break
            
            if conflicts:
                st.warning("⚠️ Potential topic-stance conflicts detected:")
                for conflict in conflicts:
                    st.write(f"- {conflict}")
                st.write("These conflicts may affect classification accuracy.")
        
        # Chunk details
        st.write("### Chunk Details")
        
        chunk_df = pd.DataFrame(chunk_details)
        if not chunk_df.empty:
            st.dataframe(chunk_df, use_container_width=True)
            
            # Sample chunks
            st.write("### Sample Chunks:")
            for i, chunk in enumerate(chunk_details[:min(3, len(chunk_details))]):
                with st.expander(f"Chunk {i+1} ({chunk['position']}) - {chunk['prediction']}"):
                    st.write(chunk["text"])
    
    # Show source content for URLs
    if isinstance(input_data, dict) and "text" in input_data:
        with st.expander("View Scraped Content"):
            st.markdown(f"**Title:** {input_data.get('title', 'No title')}")
            content_preview = input_data.get('text', 'No content')
            st.markdown("**Content:**")
            st.markdown(content_preview[:1000] + "..." if len(content_preview) > 1000 else content_preview)

# --- Main Streamlit App ---
def main():
    st.title("📰 Enhanced Political Leaning Classifier")
    st.write("Classify news articles as **Left**, **Center**, or **Right** leaning with advanced topic-stance analysis.")
    
    # Display info about loading the model
    st.info("This application requires the transformers library and a pre-trained DeBERTa model.")
    
    # Check if transformers is installed
    try:
        # Only import when needed to avoid crashing if not installed
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        transformers_available = True
    except ImportError:
        transformers_available = False
        st.error("The transformers library is not installed. Please install it using: pip install transformers")
        st.write("You can continue using the app without classification functionality.")
    
    # Try to import the interpretation library - not critical
    try:
        from transformers_interpret import SequenceClassificationExplainer
        interpret_available = True
    except ImportError:
        interpret_available = False
        st.warning("The transformers_interpret library is not installed. Word attribution will not be available.")
    
    # Input options
    input_type = st.radio("Choose input method:", ["Paste Text", "URL"])
    
    # Advanced options
    with st.expander("Advanced Options"):
        extraction_method = st.selectbox(
            "Content extraction method (for URLs):",
            ["Auto (fastest successful method)", "BeautifulSoup", "Newspaper3k", "Trafilatura"],
            index=0
        )
        
        # Only show attribution option if the library is available
        generate_attribution = False
        if interpret_available:
            generate_attribution = st.checkbox("Generate word attribution (slower)", value=False)
        
        viz_type = st.radio(
            "Visualization detail level:",
            ["Basic", "Detailed"]
        )
        
        # Add model path configuration option
        model_path = st.text_input(
            "Model path (leave empty to use demo mode):",
            value=""
        )
    
    # Load model only if needed and if transformers is available
    model = None
    tokenizer = None
    explainer = None
    
    if transformers_available and (
        (input_type == "Paste Text" and st.button("Classify Text")) or
        (input_type == "URL" and st.button("Scrape & Classify URL"))
    ):
        # Try to load the model
        with st.spinner("Loading model..."):
            try:
                model, tokenizer = load_model(model_path)
                if interpret_available and generate_attribution:
                    explainer = SequenceClassificationExplainer(model, tokenizer)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.error("Please check if the model path is correct or try running in demo mode.")
                return
        
        # Process the input
        if input_type == "Paste Text":
            user_input = st.session_state.get('text_input', '')
            if not user_input.strip():
                st.warning("Please enter some text.")
                return
                
            with st.spinner("⏳ Analyzing text..."):
                process_text_input(user_input, model, tokenizer, explainer, generate_attribution, viz_type)
                
        else:  # URL input
            url = st.session_state.get('url_input', '')
            if not url.strip():
                st.warning("Please enter a URL.")
                return
                
            with st.spinner("⏳ Scraping content..."):
                method_param = extraction_method.lower().split(" ")[0] if extraction_method != "Auto (fastest successful method)" else "auto"
                process_url_input(url, method_param, model, tokenizer, explainer, generate_attribution, viz_type)
    
    # Input fields - using session_state to maintain values after button clicks
    if input_type == "Paste Text":
        if 'text_input' not in st.session_state:
            st.session_state['text_input'] = ""
        
        st.text_area(
            "Paste your article text below:", 
            height=250,
            key="text_input"
        )
        
        if not transformers_available:
            st.button("Classify Text", disabled=True)
            
    else:  # URL input
        if 'url_input' not in st.session_state:
            st.session_state['url_input'] = ""
            
        st.text_input(
            "Enter the article URL:",
            key="url_input"
        )
        
        if not transformers_available:
            st.button("Scrape & Classify URL", disabled=True)

if __name__ == "__main__":
    main()