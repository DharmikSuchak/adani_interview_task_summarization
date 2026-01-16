# Text Summarization using LLM

## Model Selection

I'm using **Azure OpenAI GPT-4o** for the final summarization step.

Why GPT-4o?
- works great for summarization out of the box, no fine-tuning needed
- no GPU required on my machine since its API based
- handles long texts well and gives good quality summaries

I also looked at `facebook/bart-large-cnn` and `google/pegasus-xsum` but they need GPU to run fast locally. GPT-4o via API is simpler.

## Data Preprocessing

For preprocessing I used these libraries:

- `re` (regex) - for cleaning text (removing URLs, HTML, extra whitespace etc). chose this over nltk/spacy because its built-in and fast, no extra dependencies needed
- `collections.Counter` - for word frequency scoring. simpler than sklearn TfidfVectorizer and works fine for what we need
- `datasets` from HuggingFace - to load the XSUM dataset. better than manual download because it handles caching and streaming

## Training and Inference Pipelines

The summarization works like this:

```
Input Article

clean_text() - remove noise, links, html

split_into_chunks() - break into ~500 token chunks

split_sentences() - get individual sentences

score_sentences() - score by word frequency + position

select_top_sentences() - pick top 2 per chunk (extractive)

rewrite_with_llm() - LLM rewrites into final summary (abstractive)

Output Summary
```

basically you just call `summarize_text(text, max_words)` and all this happens behind the scenes.

## Frontend

I wanted to keep the frontend simple and fast so I used:
- Flask to serve the HTML template
- TailwindCSS via CDN for styling (no build step needed)
- vanilla JavaScript for the API calls

no React or complex frameworks, just a simple textarea, a button, and some sample articles to try. keeps things lightweight.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

then open http://localhost:5000

## Dataset

using XSUM dataset from HuggingFace (EdinburghNLP/xsum)

```python
from datasets import load_dataset
ds = load_dataset("EdinburghNLP/xsum")
```
