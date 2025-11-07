# 🧠 RAG-Based Research Paper Analysis Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for automated **research paper understanding, evaluation, and impact estimation** using LLMs and open scholarly APIs.  
This project processes academic PDFs end-to-end — from text extraction to computing a **Composite Impact and Relevance (CIR)** score.

---

## 🎥 Demo

Watch the full demonstration on Google Drive:  
🔗 [View Demo](https://drive.google.com/file/d/13ycBf2izNtGNOt0qXHfGTjO6w8fqxnwa/view?usp=drive_link)

---

## 🚀 Features

### 1. PDF Text Extraction
- Downloads and extracts text from research papers (e.g., arXiv links) using **pdfplumber**.  
- Handles both local and remote PDFs.

### 2. LLM-Based Research Analysis
- Summarizes papers into structured sections:
  - **Title**
  - **Authors**
  - **Abstract**
  - **Key Concepts**
  - **Methodology**
  - **Main Findings**
- Uses the **Groq LLM API (llama-3.3-70b-versatile)** for deep text understanding.

### 3. Chunk Generation for RAG
- Splits extracted text into context-aware chunks suitable for retrieval or claim validation.

### 4. UCR (Unsupported Claim Rate) Evaluation
- Assesses factual consistency between generated summaries and the source document.

### 5. CIR (Composite Impact & Relevance) Estimation
- Estimates paper impact and novelty using:
  - **Semantic Scholar** (citations)
  - **CrossRef** (fallback source)
  - **OpenAlex** (field normalization)
  - **Groq LLM** (novelty estimation)
- Outputs normalized **Impact** and **Relevance** metrics.

---

## 🧩 Dependencies

Install all required Python packages:
```bash
pip install pdfplumber groq pydantic requests tqdm openalexapi
