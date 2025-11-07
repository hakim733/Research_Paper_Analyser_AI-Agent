import os, io, re, json, requests, pdfplumber, math
from typing import List, Dict, Any
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# ====== LLM BACKEND ======
class LLMBackend:
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, prompt, system=None, temperature=0.2):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        return resp.choices[0].message.content.strip()

# ====== PAPER PROCESSOR ======
class RAGResearchProcessorLLM:
    def __init__(self, llm: LLMBackend):
        self.llm = llm
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_document_text(self, url):
        r = requests.get(url)
        pdf_file = io.BytesIO(r.content)
        full_text, pages = "", []
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                full_text += text + "\n"
                pages.append({"page_number": i + 1, "content": text})
        return full_text, pages

    def analyze_research_paper(self, text):
        prompt = f"""
        Extract this research paper info as JSON:
        {{ "title": "", "authors": [], "abstract": "", 
           "key_concepts": [], "methodology": "", "main_findings": [] }}
        --- TEXT ---
        {text[:8000]}
        """
        raw = self.llm.chat(prompt)
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group()) if m else {}
        except:
            return {}

    def create_rag_chunks(self, pages, chunk_size=500):
        chunks = []
        for p in pages:
            text = p["content"]
            if not text.strip(): continue
            sentences = re.split(r"[.!?]+", text)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) < chunk_size:
                    buf += s + ". "
                else:
                    chunks.append({"page": p["page_number"], "content": buf})
                    buf = s + ". "
            if buf: chunks.append({"page": p["page_number"], "content": buf})
        return chunks

    def retrieve_relevant_chunks(self, query, chunks, top_k=3):
        """Return top_k text chunks most similar to query."""
        # Handle both dict format (with "content" key) and string format
        texts = []
        for c in chunks:
            if isinstance(c, dict):
                texts.append(c.get("content", c.get("text", str(c))))
            else:
                texts.append(str(c))
        
        q_emb = self.embedder.encode([query])
        c_embs = self.embedder.encode(texts)
        sims = cosine_similarity(q_emb, c_embs)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [{"text": texts[i], "score": float(sims[i])} for i in top_idx]

# ====== CIR ESTIMATOR ======
class CIREstimator:
    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def fetch_citations(self, title: str):
        try:
            api = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1&fields=citationCount"
            r = requests.get(api, timeout=15)
            data = r.json().get("data", [])
            return data[0].get("citationCount", 0) if data else 0
        except:
            return 0

    def estimate_novelty(self, abstract: str):
        prompt = f"Rate novelty 0-1 as JSON: {{'novelty': value}} Abstract: {abstract}"
        try:
            raw = self.llm.chat(prompt)
            m = re.search(r"\{.*\}", raw, re.S)
            return json.loads(m.group()).get("novelty", 0.5)
        except:
            return 0.5

    def compute_cir(self, title: str, abstract: str):
        cit = self.fetch_citations(title)
        novelty = self.estimate_novelty(abstract)
        norm_cit = min(1.0, cit / 100)
        cir = 0.5 * norm_cit + 0.5 * novelty
        return {"citations": cit, "novelty": round(novelty, 2), "CIR": round(cir, 2)}

# ====== CLAIM SUPPORT ======
class LLMUCREvaluator:
    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def analyze_claim_support(self, text: str, chunks: List[Dict]):
        claims = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.split()) > 5]
        ctx = "\n\n".join(ch["content"] for ch in chunks[:5])
        supported, unsupported = [], []
        for c in claims:
            prompt = f"Claim: {c}\nContext: {ctx}\nIs it supported? (SUPPORTED/UNSUPPORTED)"
            r = self.llm.chat(prompt)
            if "SUPPORTED" in r.upper():
                supported.append(c)
            else:
                unsupported.append(c)
        total = len(claims)
        return {
            "total": total,
            "supported": len(supported),
            "unsupported": len(unsupported),
            "UCR": len(unsupported) / total if total else 0
        }

