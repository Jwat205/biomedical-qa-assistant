# 🧬 Biomedical QA Assistant

This is a Streamlit-based AI assistant that answers biomedical questions by searching and summarizing scientific papers. It uses:

- 🔍 FAISS + BM25 for hybrid semantic retrieval
- 🧠 BioBERT for biomedical embeddings
- 💬 Perplexity AI or ChatGPT for fluent, citation-based answers
- 📄 Google Cloud Storage for hosting PDFs
- 📚 APA citation formatting

## 🚀 Usage

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔐 Secrets Setup

- Set your Google Cloud credentials:
  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS="your-service-account.json"
  ```
- Add your Perplexity or OpenAI key directly in the code or use `.env` and `python-dotenv`

## 🧪 Example PDFs

Upload biomedical PDFs to your GCS bucket and update the filenames in `app.py`.

## 📤 Exports

Answers and sources are exported to `qa_export.csv` with APA citations.

---

Built by [Your Name], 2025.
