# GenAI Customer Support Chatbot — Healthcare KPI Assistant

> **✅ Repository Status: Fully implemented and runnable. See Quick Start below.**

---

A production-grade RAG (Retrieval-Augmented Generation) chatbot that answers natural language questions about doctor-specific KPIs and clinic performance metrics.

Built to mirror a real deployment at a healthcare operations company — replacing manual Power BI dashboard lookups with an AI-powered conversational interface.

---

## Architecture

```
User Question
     │
     ▼
[Streamlit UI] ──── HTTP ────► [FastAPI Backend]
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                   [FAISS Vector Store]    [OpenAI GPT-4o]
                   (KPI embeddings)        (Response generation)
                          │
                          ▼
                   Top-K relevant
                   KPI documents
                          │
                          └──► Augmented prompt ──► Answer
```

**Production equivalent:** Azure AI Search → FAISS | Azure OpenAI → OpenAI API

---

## Features

- Natural language queries over structured KPI data
- Multi-turn conversation with context retention
- Doctor-specific and department-level filtering via semantic search
- FastAPI REST backend — integrates with Power BI, dashboards, or any frontend
- Streamlit UI with source attribution and latency tracking
- Docker + docker-compose for one-command deployment

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/akashsrinivasan12/genai-customer-support-chatbot
cd genai-customer-support-chatbot

cp .env.example .env
# Add your OpenAI API key to .env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the vector index

```bash
python -c "from src.vector_store import build_index; build_index()"
```

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

### 5. Run the Streamlit UI (new terminal)

```bash
streamlit run app/streamlit_app.py
```

### Or run everything with Docker

```bash
docker-compose up --build
```

---

## API Reference

### `POST /chat`

```json
{
  "question": "What is Dr. Smith's patient wait time this month?",
  "top_k": 3,
  "chat_history": []
}
```

**Response:**
```json
{
  "answer": "Dr. Smith's patient wait time for May 2026 is 12 minutes, down 18% from last month...",
  "sources": ["Dr. Smith — Patient Wait Time (May 2026)"],
  "latency_ms": 843.2,
  "tokens_used": 187
}
```

### `POST /rebuild-index`
Rebuilds the FAISS vector index from the latest `data/healthcare_kpis.json`.

---

## Example Questions

- *"What is Dr. Smith's patient wait time this month?"*
- *"Which doctor has the highest satisfaction score?"*
- *"What's the clinic's total revenue for May 2026?"*
- *"How did the no-show rate change for Dr. Patel?"*
- *"What's the appointment utilization rate?"*

---

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | FAISS (production: Azure AI Search) |
| LLM | OpenAI GPT-4o-mini (production: Azure OpenAI) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + docker-compose |

---

## Project Structure

```
genai-customer-support-chatbot/
├── api/
│   └── main.py          # FastAPI endpoints
├── app/
│   └── streamlit_app.py # Chat UI
├── src/
│   ├── vector_store.py  # FAISS indexing + search
│   └── rag_pipeline.py  # Retrieval + generation
├── data/
│   └── healthcare_kpis.json  # Synthetic KPI dataset
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Author

**Akash Srinivasan** — [LinkedIn](https://linkedin.com/in/akashsrinivasan12) | [GitHub](https://github.com/akashsrinivasan12)
