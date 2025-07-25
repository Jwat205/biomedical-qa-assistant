# First, let's just improve your README.md with better documentation
# This is 100% safe and won't break anything

# Step 1: Just update your README.md with this content:

"""
# 🧬 Biomedical QA Assistant

AI-powered Streamlit assistant for biomedical research using BioBERT + Perplexity + FAISS.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Google Cloud credentials:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="your-service-account.json"
   ```

3. **Add your API keys:**
   - Create a `.env` file with your Perplexity or OpenAI key
   - Or add them directly in the code

4. **Update PDF filenames:**
   - Upload biomedical PDFs to your GCS bucket
   - Update the filenames in `app.py`

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## 📊 Features

- 🔍 FAISS + BM25 hybrid semantic retrieval
- 🧠 BioBERT for biomedical embeddings  
- 💬 Perplexity AI or ChatGPT for fluent answers
- 📄 Google Cloud Storage for PDF hosting
- 📚 APA citation formatting
- 📥 Export answers to CSV

## 📁 Current Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `qa_export.csv` - Generated Q&A exports (created when you export)

## ⚙️ Configuration

Create a `.env` file:
```
PERPLEXITY_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## 🤝 Contributing

Built by Jesse Watson, 2025. Feel free to contribute!

## 📞 Support

For issues or questions, please open a GitHub issue.
"""

# Step 2: Add this .env.example file (completely safe):
"""
# Copy this to .env and fill in your actual keys
PERPLEXITY_API_KEY=your_perplexity_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
"""

# Step 3: Add this .gitignore file (safe - just prevents committing sensitive files):
"""
# Environment variables
.env

# Python cache
__pycache__/
*.pyc

# Streamlit
.streamlit/

# Google Cloud credentials
*.json
service-account*.json

# Data exports
*.csv
exports/

# IDE files
.vscode/
.idea/
"""
