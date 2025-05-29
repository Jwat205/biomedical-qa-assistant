 import streamlit as st
from streamlit.components.v1 import html
import fitz  # PyMuPDF
import re
import torch
import faiss
import numpy as np
import os
import requests
import nltk
from tqdm import tqdm
from nltk.corpus import wordnet
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from google.cloud import storage
from torch.nn.functional import cosine_similarity
from collections import defaultdict
import difflib
import csv

# Page setup
theme_color = "#0077B6"
st.set_page_config(page_title="Biomedical QA", layout="wide")

st.markdown(f"""
    <style>
    .block-container {{
        padding: 2rem 3rem;
    }}
    .stTextInput > div > div > input {{
        border: 2px solid {theme_color};
        border-radius: 12px;
        padding: 10px;
    }}
    .stButton button {{
        background-color: {theme_color};
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }}
    .stSubheader {{
        color: {theme_color};
        font-size: 1.3rem;
    }}
    .highlight-answer {{
        background-color: #E0F7FA;
        border-left: 5px solid {theme_color};
        padding: 1rem;
        border-radius: 6px;
    }}
    </style>
""", unsafe_allow_html=True)

nltk.download("wordnet")

# BioBERT setup
model_id = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# Perplexity API setup
PPLX_API_KEY = "HIDDEN"
PPLX_API_URL = "HIDDEN"
PPLX_MODEL = "sonar-pro"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {PPLX_API_KEY}"
}

citation_log = []

# PDF loading
@st.cache_data
def load_pdf_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    pdf_bytes = blob.download_as_bytes()
    doc = fitz.open("pdf", pdf_bytes)
    return "\n".join(page.get_text("text") for page in doc)

# Chunking
@st.cache_data
def split_text_into_chunks(text, chunk_size=10, overlap=3):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return [" ".join(cleaned[i:i + chunk_size]) for i in range(0, len(cleaned), chunk_size - overlap)]

@st.cache_data
def load_all_pdfs_and_chunks(bucket_name, pdf_filenames):
    all_chunks = []
    for filename in pdf_filenames:
        text = load_pdf_from_gcs(bucket_name, filename)
        chunks = split_text_into_chunks(text)
        all_chunks.extend([{ "text": c, "source": filename } for c in chunks])
    return all_chunks

# Embedding
@st.cache_data
def get_biomedical_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

@st.cache_data
def store_embeddings_in_faiss(chunks, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = np.vstack([get_biomedical_embedding(text) for text in texts])
    faiss.normalize_L2(embeddings)
    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.add(embeddings)
    sources = [chunk["source"] for chunk in chunks]
    return index, texts, sources

# Retrieval
def expand_query(query):
    words = query.split()
    expanded_words = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            expanded_words.update(lemma.name() for lemma in syn.lemmas())
    return " ".join(expanded_words)

def retrieve_best_chunk(query, index, texts, sources):
    text_to_source = {text: source for text, source in zip(texts, sources)}
    doc_to_chunks = defaultdict(list)
    expanded_queries = [query, query.lower(), expand_query(query)]

    for q in expanded_queries:
        query_embedding = get_biomedical_embedding(q).reshape(1, -1)
        distances, indices = index.search(query_embedding, k=5)
        for i in indices[0]:
            doc_to_chunks[text_to_source[texts[i]]].append(texts[i])

    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    bm25_scores = bm25.get_scores(query.split())

    for idx in np.argsort(bm25_scores)[-3:][::-1]:
        doc_to_chunks[text_to_source[texts[idx]]].append(texts[idx])

    best_doc = max(doc_to_chunks.items(), key=lambda kv: len(kv[1]))[0]

    query_embedding = get_biomedical_embedding(query)
    top_chunks = sorted(
        doc_to_chunks[best_doc],
        key=lambda chunk: cosine_similarity(
            torch.tensor(query_embedding),
            torch.tensor(get_biomedical_embedding(chunk))
        ).item(),
        reverse=True
    )[:3]

    return top_chunks, best_doc

def find_closest_source(retrieved_text, all_chunks):
    best_match = max(all_chunks, key=lambda chunk: difflib.SequenceMatcher(None, retrieved_text, chunk["text"]).ratio(), default={"source": "unknown"})
    return best_match["source"]

def log_citation(source):
    citation_log.append(source)

def export_to_csv(results, filename="qa_export.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Chunk", "Answer", "Source", "APA"])
        for q, chunk, ans, src in results:
            apa = f"({src.split('/')[-1].replace('-', ' ').replace('.pdf','')})"
            writer.writerow([q, chunk, ans, src, apa])

def ask_perplexity_with_citation(query, context, all_filenames):
    trimmed_context = " ".join(context.split()[:600])
    file_list = "\n".join(all_filenames)

    prompt = f"""
    You are a biomedical expert and language editor. Given the raw extracted biomedical context below, your task is twofold:
    1. First, clean and organize the retrieved text into coherent bullet points, paragraphs, or sections so it's easier to read.
    2. Second, use the organized context to answer the biomedical question.
    3. Finally, infer and list the most likely source document in APA format, selecting from the filenames listed.

    --- Context Start ---
    {trimmed_context}
    --- Context End ---

    Question: {query}

    List of PDFs:
    {file_list}

    Please respond in this format:
    ## 📁 Organized Context
    [Formatted version here]

    ## 🧐 Answer
    [Concise and complete answer]

    ## 📚 APA Citation
    [Most likely source, APA format]
    """

    data = {
            "model": PPLX_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 3000,
            "temperature": 0.2,
        }

    response = requests.post(PPLX_API_URL, headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else "API error."

# Load data
BUCKET_NAME = "pdf-ai-storage"
filenames = [
        "KweHealth Data Set/Amniotic Fluid/Bossolasco-Molecular and phenotypic characterization of human amniotic fluid cells and their differentiation potential-2006-Cell Research.pdf",
        "KweHealth Data Set/Amniotic Fluid/Bowen-Cell-Free Amniotic Fluid and Regenerative Medicine- Current Applications and Future Opportunities-2022-Biomedicines.pdf",
        "KweHealth Data Set/Amniotic Fluid/Chanda-Acellular Human Amniotic Fluid-Derived Extracellular Vesicles as Novel Anti-Inflammatory Therapeutics against SARS-CoV-2 Infection-2024-Viruses.pdf",
        "KweHealth Data Set/Amniotic Fluid/Dixon-Amniotic Fluid Exosome Proteomic Profile Exhibits Unique Pathways of Term and Preterm Labor-2018-Endocrinology.pdf",
        "KweHealth Data Set/Amniotic Fluid/Dusza-Identification of known and novel nonpolar endocrine disruptors in human amniotic fluid-2022-Environment International.pdf",
        "KweHealth Data Set/Amniotic Fluid/Dusza-Method Development for Effect-Directed Analysis of Endocrine Disrupting Compounds in Human Amniotic Fluid-2019-Environmental Science & Technology.pdf",
        "KweHealth Data Set/Amniotic Fluid/Gholizadeh-Ghalehaziz-A Mini Overview of Isolation, Characterization and Application of Amniotic Fluid Stem Cells-2015-International Journal of Stem Cells.pdf",
        "KweHealth Data Set/Amniotic Fluid/Han-Single‐vesicle imaging and co‐localization analysis for tetraspanin profiling of individual extracellular vesicles-2021-Journal of Extracellular Vesicles.pdf",
        "KweHealth Data Set/Amniotic Fluid/Katsikantami-Phthalate metabolites concentrations in amniotic fluid and maternal urine- Cumulative exposure and risk assessment-2020-Toxicology Reports.pdf",
        "KweHealth Data Set/Amniotic Fluid/Orczyk-Pawilowicz-Metabolomics of Human Amniotic Fluid and Maternal Plasma during Normal Pregnancy-2016-PLoS ONE.pdf",
        "KweHealth Data Set/Amniotic Fluid/Priščáková-Syncytin-1, syncytin-2 and suppressyn in human health and disease-2023-Journal of Molecular Medicine.pdf",
        "KweHealth Data Set/Amniotic Fluid/Protzman-Placental-Derived Biomaterials and Their Application to Wound Healing- A Review-2023-Bioengineering.pdf",
        "KweHealth Data Set/Amniotic Fluid/Rivero-Human amniotic fluid derived extracellular vesicles attenuate T cell immune response-2022-Frontiers in Immunology.pdf",
        "KweHealth Data Set/Amniotic Fluid/Shamsnajafabadi-Amniotic fluid characteristics and its application in stem cell therapy- A review-2022-International Journal of Reproductive Biomedicine.pdf"
         ]
  # same as before
all_chunks = load_all_pdfs_and_chunks(BUCKET_NAME, filenames)
index, texts, sources = store_embeddings_in_faiss(all_chunks)

# Main UI
st.title("🧬 Biomedical Research Assistant")
st.markdown("""
Welcome to the **Biomedical Chatbot**. This assistant helps you:
- Search through medical literature using **advanced AI**
- Understand answers with **structured, clean summaries**
- Get **APA-style citations** for research use
""")

query = st.text_input("💡 Ask your biomedical question:")
run_button = st.button("🔍 Search")

results = []
if run_button and query:
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        top_chunks, _ = retrieve_best_chunk(query, index, texts, sources)
        context = " ".join(top_chunks)
        correct_source = find_closest_source(top_chunks[0], all_chunks)
        log_citation(correct_source)

        answer = ask_perplexity_with_citation(query, context, filenames)

        st.subheader("📌 Retrieved Text")
        st.markdown(f"<div class='highlight-answer'>{context}</div>", unsafe_allow_html=True)

        st.subheader("🚀 AI Answer")
        st.markdown(f"<div class='highlight-answer'>{answer}</div>", unsafe_allow_html=True)

        results.append((query, context, answer, correct_source))
        export_to_csv(results)

st.markdown("""
---
🧠 Powered by **BioBERT**, **FAISS**, **BM25**, and **Perplexity AI**. Built for the future of **biomedical research**. 
""")
