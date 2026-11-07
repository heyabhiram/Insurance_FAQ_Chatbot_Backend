# 🚗 AI-Powered Vehicle Insurance FAQ Chatbot Backend

A production-ready, modular, and beginner-friendly backend for an AI-powered Vehicle Insurance FAQ Chatbot. Built with **FastAPI**, **Sentence Transformers**, **ChromaDB**, **LangChain**, and **TCS GenAI Lab LLM** (`genailab-maas-gpt-4o`).

---

## 📋 Table of Contents
1. [What the Project Does](#-what-the-project-does)
2. [How the Chatbot Works](#-how-the-chatbot-works)
3. [Technologies Used and Why](#-technologies-used-and-why)
4. [Project Folder Structure](#-project-folder-structure)
5. [Explanation of Important Python Files](#-explanation-of-important-python-files)
6. [How the FAQ Dataset is Structured](#-how-the-faq-dataset-is-structured)
7. [How Embeddings Work (`all-MiniLM-L6-v2`)](#-how-embeddings-work-all-minilm-l6-v2)
8. [How ChromaDB Stores and Retrieves FAQs](#-how-chromadb-stores-and-retrieves-faqs)
9. [Step-by-Step RAG Pipeline](#-step-by-step-rag-pipeline)
10. [TCS GenAI Lab LLM Integration](#-tcs-genai-lab-llm-integration)
11. [Intent Recognition & Entity Extraction](#-intent-recognition--entity-extraction)
12. [Multi-Turn Conversation Context](#-multi-turn-conversation-context)
13. [Installation & Setup Guide](#-installation--setup-guide)
14. [How to Ingest FAQs](#-how-to-ingest-faqs)
15. [How to Run the FastAPI Server](#-how-to-run-the-fastapi-server)
16. [Interactive Swagger API Documentation](#-interactive-swagger-api-documentation)
17. [Testing the Chatbot with Examples](#-testing-the-chatbot-with-examples)
18. [Common Errors & Troubleshooting](#-common-errors--troubleshooting)

---

## 🎯 What the Project Does

The **Vehicle Insurance FAQ Chatbot** assists vehicle owners and policyholders by answering motor insurance questions in natural language. It covers real-world Indian motor insurance regulations and practices, including:
- **Comprehensive & Third-Party Insurance** (differences, requirements, legal mandates)
- **Standalone Own-Damage (OD) Coverage** (accidental damages, fire, flood, natural calamities)
- **Insured Declared Value (IDV)** (definition, depreciation slabs, premium impact)
- **No Claim Bonus (NCB)** (discount slabs from 20% to 50%, transfer procedures, NCB Protect)
- **Claim Procedures** (accidental claims, vehicle theft, cashless vs. reimbursement, required documents)
- **Policy Renewals & Expirations** (90-day grace period, inspections, online renewals)
- **Policy Endorsements** (adding CNG/LPG kits, updating name/address, adding/removing bank hypothecation)
- **Deductibles & Exclusions** (compulsory vs. voluntary deductibles, drunk driving, driving without licence)
- **Add-on Covers** (Zero Depreciation, Roadside Assistance, Engine Protect, Return to Invoice, CPA)

> **Important Safety Boundary:** This chatbot is an educational FAQ assistant. It never fabricates customer policy records, personal premium calculations, claim approvals, or insurer-specific conditions.

---

## 🧠 How the Chatbot Works

1. **User asks a question** (e.g., *"What is IDV and how is it calculated?"*).
2. **Text Normalization**: Strips excess whitespace, standardizes punctuation, and expands contractions.
3. **NLP Processing**:
   - Classifies user intent (e.g., `idv`).
   - Extracts insurance entities (e.g., vehicle type, claim number, IDV amount).
4. **Context Enrichment**: If the user asks a follow-up (e.g., *"How does it affect my premium?"*), the system pairs the prior context with the current query.
5. **Vector Search (ChromaDB)**: Encodes query using `sentence-transformers/all-MiniLM-L6-v2` and fetches the top 3 most relevant FAQ entries.
6. **Relevance Threshold Check**: Discards results if similarity falls below the configured threshold (`SIMILARITY_THRESHOLD = 0.35`).
7. **RAG Response Generation (TCS GenAI Lab LLM)**: Passes the factual FAQ context into `genailab-maas-gpt-4o` to generate a polite, customer-friendly answer. If the LLM is unconfigured or unavailable, the system safely falls back to the verified FAQ answer.

---

## 🛠️ Technologies Used and Why

| Technology | Role | Why It Was Chosen |
|---|---|---|
| **Python 3.11+** | Runtime environment | Modern Python with enhanced performance and strong type hinting. |
| **FastAPI** | Web Framework | Extremely fast, asynchronous, automatic validation with Pydantic, and automatic Swagger docs. |
| **Pydantic v2** | Data Validation | Ensures clean schemas for requests, responses, and environment variables. |
| **Sentence Transformers** | Embedding Engine | Uses `sentence-transformers/all-MiniLM-L6-v2`, an efficient 384-dimensional model with high semantic search accuracy. |
| **ChromaDB** | Vector Database | Lightweight, embedded vector store that persists embeddings locally without needing external database clusters. |
| **LangChain & langchain-openai** | LLM Orchestration | Standardized interface for prompt engineering, message history, and OpenAI-compatible endpoint calls. |
| **HTTPX** | HTTP Client | Async/sync HTTP client supporting custom SSL configuration for corporate enterprise networks. |
| **SQLite3** | History Storage | Lightweight, zero-config relational store for multi-turn chat threads. |
| **Pytest** | Testing Suite | Industry standard testing framework for automated unit and integration tests. |

---

## 📁 Project Folder Structure

```text
backend/
├── app/
│   ├── __init__.py               # Marks app directory as a Python package
│   ├── main.py                   # FastAPI application entrypoint, CORS, lifespan, and /health
│   ├── config.py                 # Central configuration loaded from .env
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat.py               # Endpoints: POST /api/v1/chat, GET /api/v1/faqs
│   ├── services/
│   │   ├── __init__.py
│   │   ├── nlp_service.py        # Rule-based Intent Recognition & Entity Extraction
│   │   ├── embedding_service.py  # SentenceTransformer all-MiniLM-L6-v2 vectorizer
│   │   ├── retrieval_service.py  # ChromaDB persistent collection & similarity search
│   │   ├── llm_service.py        # TCS GenAI Lab ChatOpenAI client & RAG generation
│   │   └── history_service.py    # SQLite-backed multi-turn conversation storage
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py      # Query cleaning and text normalization
├── data/
│   └── vehicle_insurance_faqs.json # 55 comprehensive, verified synthetic FAQs
├── chroma_db/                    # Local ChromaDB vector database files (generated)
├── scripts/
│   ├── __init__.py
│   └── ingest_faqs.py            # CLI script to embed and load FAQs into ChromaDB
├── tests/
│   ├── __init__.py
│   ├── test_chat.py              # API endpoint and health check tests
│   └── test_retrieval.py         # Dataset, NLP, and retrieval unit tests
├── .env                          # Local environment secrets (not committed)
├── .env.example                  # Environment template
├── .gitignore                    # Ignores venv, db, and cache files
├── requirements.txt              # Pinned Python package dependencies
├── run.txt                       # Quick copy-paste running commands
└── README.md                     # Comprehensive documentation
```

---

## 🔍 Explanation of Important Python Files

### 1. `app/config.py`
Defines the `Settings` class using Pydantic's `BaseSettings`. Loads variables from `.env`:
- `TCS_API_KEY`: API key for TCS GenAI Lab.
- `TCS_BASE_URL`: Endpoint (`https://genailab.tcs.in`).
- `TCS_MODEL`: Model name (`genailab-maas-gpt-4o`).
- `TCS_VERIFY_SSL`: Configurable TLS verification (default: `False` for lab networks).
- `EMBEDDING_MODEL`: Pinned to `"sentence-transformers/all-MiniLM-L6-v2"`.
- `SIMILARITY_THRESHOLD`: Retrieval cutoff (0.35).

### 2. `app/services/embedding_service.py`
Wraps `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`. Provides:
- `embed_text(text)`: Converts a query string into a 384-dimensional float vector.
- `embed_texts(texts)`: Efficient batch encoding for FAQ dataset ingestion.

### 3. `app/services/retrieval_service.py`
Connects to local `ChromaDB` with `hnsw:space="cosine"`.
- `ingest_faqs()`: Reads `vehicle_insurance_faqs.json`, embeds documents, and indexes them.
- `retrieve(query, top_k=3)`: Calculates cosine similarity `(1 - distance)` and returns the top 3 items matching the threshold.
- `get_all_faqs(category)`: Returns raw FAQ entries for catalog viewing.

### 4. `app/services/nlp_service.py`
Provides rule-based Natural Language Processing:
- **Intents**: `policy_coverage`, `third_party_insurance`, `comprehensive_insurance`, `claim_process`, `claim_status`, `premium_payment`, `policy_renewal`, `endorsement`, `ncb`, `idv`, `exclusions`, `eligibility`, `general_faq`, `unknown`.
- **Entities**: Extracts `vehicle_type`, `policy_type`, `claim_number`, `policy_number`, `coverage_type`, `renewal_date`, and `idv_amount`.

### 5. `app/services/llm_service.py`
Integrates with TCS GenAI Lab using LangChain:
```python
client = httpx.Client(verify=settings.TCS_VERIFY_SSL)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="genailab-maas-gpt-4o",
    api_key=os.getenv("TCS_API_KEY"),
    http_client=client
)
```
- Restricts responses strictly to retrieved FAQ context.
- Falls back safely if the LLM key is missing or network connection fails.

### 6. `app/services/history_service.py`
Maintains conversational memory in `chat_history.db` (SQLite). Allows the assistant to understand follow-up questions referencing previous answers.

### 7. `app/routes/chat.py`
Exposes the core REST endpoints:
- `POST /api/v1/chat`: Complete RAG pipeline processing.
- `GET /api/v1/faqs`: Listing and filtering FAQs by category.

### 8. `app/main.py`
Bootstraps FastAPI, configures CORS, handles database startup checks via lifespan context, and serves `/health`.

---

## 📊 How the FAQ Dataset is Structured

The dataset file is located at `data/vehicle_insurance_faqs.json`. It contains **55 verified FAQs** covering all required topics.

Sample entry:
```json
{
  "id": "faq_005",
  "question": "What is Insured Declared Value (IDV) in vehicle insurance?",
  "answer": "Insured Declared Value (IDV) is the maximum insured sum payable by the insurance company in the event of total loss, constructive total loss, or theft of your vehicle. It represents the manufacturer's selling price minus depreciation based on the age of the vehicle.",
  "category": "idv",
  "keywords": ["idv", "insured declared value", "market value", "sum insured", "total loss"],
  "source": "Vehicle Insurance FAQ"
}
```

---

## 🧮 How Embeddings Work (`all-MiniLM-L6-v2`)

1. **What is an embedding?** An embedding is a numerical vector (a list of 384 floating-point numbers) that captures the semantic meaning of text.
2. **Why `all-MiniLM-L6-v2`?** It is optimized for semantic search. Sentences with similar meanings produce vectors that are close to each other in vector space, even if they use completely different words (e.g., *"How much will insurer pay if car is stolen?"* and *"What is Insured Declared Value?"*).
3. **Cosine Similarity**: We calculate the cosine of the angle between query vector $u$ and FAQ vector $v$:
   $$\text{similarity} = \frac{u \cdot v}{\|u\| \|v\|}$$
   A score near `1.0` means near-identical meaning; a score near `0.0` means unrelated.

---

## 📦 How ChromaDB Stores and Retrieves FAQs

- ChromaDB is an embedded vector database that runs inside your Python process.
- Each document is stored with:
  - **ID**: `faq_001`
  - **Vector**: 384-dimensional embedding generated by Sentence Transformers
  - **Metadata**: `{"question": "...", "category": "...", "source": "..."}`
  - **Document**: The raw text snippet
- When a search query arrives, ChromaDB uses an HNSW (Hierarchical Navigable Small World) index to find the nearest vectors in milliseconds.

---

## 🔄 Step-by-Step RAG Pipeline

```text
[User Question]
       │
       ▼
[1. Normalize Query] ───► lowercased, contractions expanded
       │
       ▼
[2. NLP Analysis] ──────► Intent (e.g., 'ncb') & Entities
       │
       ▼
[3. Multi-Turn Check] ──► Combines follow-up with prior question if needed
       │
       ▼
[4. SentenceTransformer]► Converts query to 384-dim vector
       │
       ▼
[5. ChromaDB Search] ───► Fetches Top-3 nearest FAQs
       │
       ├─► Score < 0.35 ──► Return Fallback: "I couldn't find sufficient verified info..."
       │
       ▼ (Score >= 0.35)
[6. Context Formulation]► Compiles verified FAQ text
       │
       ▼
[7. TCS GenAI Lab LLM] ──► Invokes ChatOpenAI (gpt-4o) with strict safety prompt
       │
       ▼
[8. SQLite History] ────► Records turn for conversational memory
       │
       ▼
[ChatResponse JSON] ────► answer, intent, entities, sources, confidence_score
```

---

## 🔐 TCS GenAI Lab LLM Integration

The chatbot connects to the TCS GenAI Lab instance using `langchain-openai`:
```python
from langchain_openai import ChatOpenAI
import httpx
import os

client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="genailab-maas-gpt-4o",
    api_key=os.getenv("TCS_API_KEY"),
    http_client=client
)
```

### Safety & Fallback Behavior:
- If `TCS_API_KEY` is not provided in `.env`, the system functions in **Offline Verified Mode**, serving the exact verified answer directly from the indexed FAQ knowledge base.
- If network connection to the lab fails, the system logs the issue and delivers the verified FAQ answer without hallucinating or breaking.

---

## 🏷️ Intent Recognition & Entity Extraction

The `NLPService` uses regex rules and terminology dictionaries:

### Detected Intents:
- `policy_coverage`: Coverage options, add-ons (Zero Dep, RSA, Engine Protect).
- `third_party_insurance`: Mandatory third-party liability rules.
- `comprehensive_insurance`: All-round comprehensive policies.
- `claim_process`: Accidental claims, theft steps, cashless vs reimbursement.
- `claim_status`: Tracking claim status, surveyor inquiries, rejection appeals.
- `premium_payment`: Payment methods, EMI, premium calculation factors.
- `policy_renewal`: Policy renewals, expired policies, break-in inspection.
- `endorsement`: CNG/LPG kit additions, name changes, hypothecation NOC.
- `ncb`: No Claim Bonus slabs (20%-50%), retention, and transfer.
- `idv`: Insured Declared Value calculations and vehicle valuation.
- `exclusions`: Mechanical breakdowns, wear and tear, drunk driving.
- `eligibility`: CPA cover, pre-owned car insurance eligibility.
- `general_faq`: Greetings, bot capabilities, help questions.
- `unknown`: Out-of-scope questions.

### Extracted Entities:
- `vehicle_type`: `car`, `bike`, `scooter`, `suv`, `commercial vehicle`, etc.
- `policy_type`: `comprehensive`, `third-party`, `standalone own damage`.
- `claim_number`: `CLM-12345`, `claim #98765`.
- `policy_number`: `POL-987654`, `policy #12345`.
- `coverage_type`: `zero depreciation`, `roadside assistance`, `rsa`.
- `renewal_date`: `15/08/2025`, `2025-10-01`.
- `idv_amount`: `5 lakhs`, `Rs. 4,50,000`.

---

## 💬 Multi-Turn Conversation Context

The chatbot stores conversation history in `chat_history.db` using SQLite.

When a user asks:
1. **User**: *"What is IDV?"*
   - **Bot**: Explains Insured Declared Value.
2. **User**: *"How does it affect my premium?"*
   - The query contains the pronoun `"it"`.
   - The chatbot detects a follow-up and enriches the search query with the previous turn: *"What is IDV? How does it affect my premium?"*.
   - ChromaDB retrieves `faq_007` (*"How does IDV affect my vehicle insurance premium?"*).
   - The answer is coherent and context-aware.

---

## 🚀 Installation & Setup Guide

### Prerequisites
- Python 3.11 or higher
- Git

### 1. Navigate to the backend folder:
```bash
cd backend
```

### 2. Create and Activate Virtual Environment (`venv`):
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Configure `.env`:
Create your `.env` file from the example:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and fill in your `TCS_API_KEY`:
```ini
TCS_API_KEY=your_tcs_genai_lab_key_here
TCS_BASE_URL=https://genailab.tcs.in
TCS_MODEL=genailab-maas-gpt-4o
TCS_VERIFY_SSL=false
```

---

## 📥 How to Ingest FAQs

Run the standalone ingestion script:
```bash
python scripts/ingest_faqs.py
```

Output:
```text
============================================================
Starting FAQ Ingestion into ChromaDB...
Data file: .../data/vehicle_insurance_faqs.json
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
============================================================
[SUCCESS] Successfully ingested 55 FAQs into ChromaDB!
ChromaDB is now primed and ready for semantic search queries.
```

*(Note: The server also auto-indexes on first launch if the vector store is empty).*

---

## ⚡ How to Run the FastAPI Server

Start the development server using Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
or run directly:
```bash
python app/main.py
```

The server starts at: `http://127.0.0.1:8000`

---

## 📖 Interactive Swagger API Documentation

FastAPI provides an automatic, interactive API testing console:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

You can test every endpoint directly inside your browser.

---

## 🧪 Testing the Chatbot with Examples

### Automated Test Suite
Run the unit and integration tests with Pytest:
```bash
pytest tests/ -v
```

### Manual Testing with cURL / Postman

#### 1. Ask a Comprehensive Insurance Question
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What is comprehensive car insurance?",
       "conversation_id": "session_01"
     }'
```
**Response:**
```json
{
  "answer": "Comprehensive vehicle insurance is an all-inclusive policy that covers damages to your own vehicle (due to accidents, fire, theft, vandalism, and natural calamities) as well as mandatory third-party liabilities...",
  "intent": "comprehensive_insurance",
  "entities": {
    "vehicle_type": "car",
    "policy_type": "comprehensive"
  },
  "sources": ["faq_001", "faq_004"],
  "confidence_score": 0.912
}
```

#### 2. Ask About IDV and Follow-Up on Premium
```bash
# Question 1
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is IDV?", "conversation_id": "session_idv"}'

# Follow-up
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How does it affect my premium?", "conversation_id": "session_idv"}'
```

#### 3. Ask an Unsupported Question (Safe Fallback)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Can you give me a recipe for chocolate cake?", "conversation_id": "session_02"}'
```
**Response:**
```json
{
  "answer": "I couldn't find sufficient verified information to answer this question. Please contact your insurance provider for assistance.",
  "intent": "unknown",
  "entities": {},
  "sources": [],
  "confidence_score": 0.0
}
```

#### 4. Filter FAQs by Category
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/faqs?category=ncb"
```

#### 5. Check Health Endpoint
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

---

## ❓ Common Errors & Troubleshooting

| Error | Root Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | Running command outside the `backend` directory. | Always run `cd backend` first, or set `PYTHONPATH=.`. |
| `SSL: CERTIFICATE_VERIFY_FAILED` when connecting to TCS GenAI Lab | Corporate network proxy / self-signed certificate. | Set `TCS_VERIFY_SSL=false` in your `.env` file. |
| `TCS_API_KEY is not set` | Empty API key in `.env`. | Add your API key in `.env`. The backend will continue to provide direct verified answers even without the key. |
| `No sentence-transformers model found / Download error` | First download of `all-MiniLM-L6-v2` requires an active internet connection. | Ensure internet access during initial model load; once downloaded, it is cached locally in `~/.cache/huggingface`. |
| Port 8000 is already in use | Another application or previous instance is running on port 8000. | Run `uvicorn app.main:app --port 8001` or terminate the existing process. |

---

*Built with ❤️ for reliable, safe, and transparent AI in Vehicle Insurance.*

