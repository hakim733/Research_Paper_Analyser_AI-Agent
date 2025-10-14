# 🧠 RAG-Based Research Paper Analysis Pipeline

## Overview
This notebook/script automates **research paper understanding, evaluation, and impact estimation** using **LLMs** and open scholarly APIs.  
It performs **end-to-end document processing** — from extracting text from PDFs to computing a *Composite Impact and Relevance (CIR)* score.

---

## 🚀 Features

### 1. **PDF Text Extraction**
- Downloads and extracts text from research papers (e.g., arXiv links) using `pdfplumber`.

### 2. **LLM-Based Research Analysis**
- Summarizes the paper into:
  - Title  
  - Authors  
  - Abstract  
  - Key Concepts  
  - Methodology  
  - Main Findings  
- Uses `Groq` LLM API (`llama-3.3-70b-versatile`).

### 3. **Chunk Generation for RAG**
- Splits extracted text into manageable context chunks for later retrieval or claim validation.

### 4. **UCR (Unsupported Claim Rate) Evaluation**
- Tests how many claims in a generated summary are supported by the source paper.

### 5. **CIR (Composite Impact & Relevance) Estimation**
- Estimates paper impact and novelty using:
  - **Semantic Scholar** for citation counts  
  - **CrossRef** as fallback  
  - **OpenAlex** for field normalization  
  - **LLM** for novelty estimation  
- Produces normalized impact and relevance metrics.

---

## 🧩 Dependencies
Install all required libraries:
```bash
!pip install pdfplumber groq pydantic requests tqdm openalexapi
```

---

## ⚙️ Configuration
Set your Groq API key before running:
```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
```

---

## ▶️ Usage

Run the main pipeline:

```python
llm = LLMBackend()
processor = RAGResearchProcessorLLM(llm)
results = processor.process_paper(url)

cir_estimator = CIREstimator(llm)
cir_results = cir_estimator.compute_realistic_cir(
    results["paper_analysis"]["title"],
    results["paper_analysis"]["abstract"]
)
```

Outputs include:
- Structured **paper summary**
- **CIR metrics** (impact, novelty, citations)
- **UCR evaluation** of claims

---

## 📊 Example Output

```
🎯 PAPER ANALYSIS SUMMARY:
{
  "title": "Attention Is All You Need",
  "authors": ["Vaswani et al."],
  "abstract": "...",
  "key_concepts": ["Transformer", "self-attention", ...]
}

📈 ESTIMATED CIR METRICS:
{
  "citations_final": 73400,
  "novelty_score": 0.9,
  "estimated_CIR": 0.95
}
```

---

## 🧪 Notes
- CIR is **heuristic**, combining LLM reasoning with citation data.
- UCR helps estimate factual alignment between generated summaries and source text.
- Designed for **RAG-For-Mobility** research analysis tasks.
