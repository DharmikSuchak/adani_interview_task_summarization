"""Text Summarization Web App
Implements: Clean -> Chunk -> Score -> Select -> Rewrite pipeline
"""
import os
from flask import Flask, request, jsonify, render_template
import requests
import re
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='templates')



LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "") # Azure OpenAI endpoint
LLM_TOKEN = os.environ.get("LLM_TOKEN", "") # Azure OpenAI API key

# Load XSUM dataset from HuggingFace
# Dataset: https://huggingface.co/datasets/EdinburghNLP/xsum
from datasets import load_dataset

def load_xsum_samples(n=5):
    """Load n samples from XSUM dataset"""
    ds = load_dataset("EdinburghNLP/xsum", split="test", streaming=True)
    samples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        samples.append({
            "id": item["id"],
            "document": item["document"],
            "summary": item["summary"]
        })
    return samples


print("Loading XSUM dataset samples...")
XSUM_SAMPLES = load_xsum_samples(5)
print(f"Loaded {len(XSUM_SAMPLES)} samples")


def clean_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\[.*?\]', '', text)  # Remove brackets content
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def split_into_chunks(text, max_tokens=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(' '.join(words[i:i+max_tokens]))
    return chunks if chunks else [text]


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def score_sentences(sentences):
    all_words = ' '.join(sentences).lower().split()
    word_freq = Counter(all_words)
    scores = []
    for i, sent in enumerate(sentences):
        words = sent.lower().split()
        freq_score = sum(word_freq[w] for w in words) / (len(words) + 1)
        position_score = 1.0 / (i + 1)  # Earlier sentences score higher
        scores.append(freq_score + position_score)
    return scores

def select_top_sentences(sentences, scores, k=3):
    if not sentences:
        return []
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in indexed[:k]])
    return [sentences[i] for i in top_indices]

def rewrite_with_llm(sentences, max_words=150):
    text = ' '.join(sentences)
    
    if not LLM_ENDPOINT or not LLM_TOKEN:
        return text
    
    headers = {"Authorization": f"Bearer {LLM_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a news summarization expert. Your task is to create clear, factual summaries that preserve key information. Focus on: WHO, WHAT, WHEN, WHERE, WHY. Avoid opinions, maintain neutral tone, and ensure the summary can stand alone without the original text."},
            {"role": "user", "content": f"Summarize the following extracted sentences into a coherent {max_words}-word news summary. Keep only the most important facts:\n\n{text}"}
        ],
        "max_tokens": 500
    }
    resp = requests.post(LLM_ENDPOINT, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def summarize_text(text, max_words=150):
    # Step 1: Clean
    text = clean_text(text)
    
    # Step 2: Chunk
    chunks = split_into_chunks(text, max_tokens=500)
    
    # Step 3: Extract top sentences from each chunk
    summary_sentences = []
    for chunk in chunks:
        sentences = split_sentences(chunk)
        if sentences:
            scores = score_sentences(sentences)
            top = select_top_sentences(sentences, scores, k=2)
            summary_sentences.extend(top)
    
    # Step 4: Rewrite using LLM (abstractive)
    if summary_sentences:
        return rewrite_with_llm(summary_sentences, max_words)
    return text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    try:
        summary = summarize_text(data['text'], data.get('max_length', 150))
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sample_articles')
def get_samples():
    return jsonify({"articles": [{"id": s["id"], "document": s["document"], "summary": s["summary"]} for s in XSUM_SAMPLES]})

@app.route('/health')
def health():
    return jsonify({"status": "ok", "llm_configured": bool(LLM_ENDPOINT and LLM_TOKEN)})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Text Summarization API")
    print("="*50)
    if not LLM_ENDPOINT or not LLM_TOKEN:
        print("\nWARNING: LLM_ENDPOINT and LLM_TOKEN not set. Set environment variables to enable LLM rewriting.")
    print("\nOpen http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
