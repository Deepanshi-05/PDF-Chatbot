# 📚 Multiple-PDF Chatbot using LLMs

Interact with multiple PDF documents **intelligently** using this AI-powered chatbot, built with 🤗 Hugging Face Transformers, LangChain, and Streamlit.

## 🚀 Features

- 📥 Upload and read multiple PDFs
- 🔍 Extracts clean, structured text from PDFs
- 🧠 Splits text into smart chunks for efficient retrieval
- 📌 Uses vector embeddings to fetch contextually relevant answers
- 🤖 + 🤖 Uses `all-MiniLM-L6-v2` for smart search and `flan-t5-base` to generate fluent answers



---

## 🛠️ Tech Stack

| Tech | Purpose |
|------|---------|
| `Streamlit` | UI and interactivity |
| `LangChain` | Chunking, pipeline management |
| `FAISS` | Semantic vector search |
| `sentence-transformers/all-MiniLM-L6-v2` | Text embedding for similarity |
| `flan-t5-base` | Natural language answer generation |
| `PyPDF2` | PDF parsing |
| `dotenv` | API token handling |

---

## 🖼️ Interface Preview

> Upload PDFs from the sidebar → Ask questions → Get intelligent, document-based answers.

```python
Question: "Summarize Chapter 3"
