# 🎓 Innopolis University AI Consultant

An advanced **Modular RAG (Retrieval-Augmented Generation)** chatbot designed for Innopolis University.
The bot answers questions about curriculum, academic rules, and legal documents using **Llama 3**, **Qdrant**, and **Cross-Encoder Re-ranking**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Aiogram](https://img.shields.io/badge/Telegram-Bot-blue)
![RAG](https://img.shields.io/badge/RAG-Modular-green)

---

## 🌟 Key Features (Grading Criteria)

This project implements an "Advanced" architecture, moving beyond naive RAG:

### 1. 🧠 Modular RAG Architecture
- **Query Transformation (Pre-Retrieval):** The system automatically detects the user's language (Regex) and injects strict system prompts. This solves the "language drift" issue where Llama 3 replies in English to Russian queries.
- **Re-ranking Module (Post-Retrieval):** Integrated `BAAI/bge-reranker-base` Cross-Encoder. It re-scores the top-k vector results to filter out noise, effectively solving the **"Lost in the Middle"** phenomenon.

### 2. 🛠 Engineering Best Practices
- **LlamaParse Integration:** Instead of standard PyPDF, I use LlamaParse to preserve complex table structures in academic curricula (PDF -> Markdown).
- **Semantic Routing:** Greetings (e.g., "Hello") are intercepted by Python logic for zero-latency responses, bypassing the expensive RAG pipeline.
- **UX Guardrails:** The bot automatically hides download buttons if the LLM admits that information was "not found".

### 3. 📊 Automated Evaluation
- Includes `benchmark.py`, an automated script that validates **Retrieval Hit Rate** against a "Golden Dataset".
- **Current Accuracy:** ~70% (Top-3 Retrieval with Reranker).

---

## 🚀 Step-by-Step Installation Guide

Follow these steps to deploy the project from scratch.

### Prerequisites
*   **Docker** & **Docker Compose** installed.
*   **Python 3.10+** (for local data ingestion).
*   **[Ollama](https://ollama.com)** installed locally (serving `llama3`).

---

### Step 1: Clone and Configure

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd atllm
    ```

2.  **Create environment file:**
    Create a `.env` file in the root directory and add your keys:

    ```env
    # Telegram Bot Token (from @BotFather)
    TELEGRAM_TOKEN=your_telegram_token_here

    # LlamaCloud API Key (for parsing PDFs)
    LLAMA_CLOUD_API_KEY=llx-your_key_here
    ```

---

### Step 2: Prepare Local Ollama

Since the bot runs in Docker but uses your local LLM, you must allow external connections to Ollama.

1.  Stop Ollama (if running).
2.  Run Ollama with host binding:
    ```bash
    OLLAMA_HOST=0.0.0.0 ollama serve
    ```
3.  **Keep this terminal open!**

---

### Step 3: Data Pipeline (ETL)

We separate heavy data processing from the bot service. Run this locally to populate the database.

1.  **Place your PDF documents** into the `data/raw/` folder.

2.  **Install local dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Parse PDFs (LlamaParse):**
    Converts PDFs to Markdown, preserving tables.
    ```bash
    python ingestion.py
    ```
    *Result: `.md` files will appear in `data/parsed/`.*

4.  **Start the Database (Qdrant):**
    We need Qdrant running to accept vectors.
    ```bash
    docker-compose up -d qdrant
    ```

5.  **Index Data (Vectorization):**
    Reads MD files, creates embeddings, and uploads them to Qdrant.
    ```bash
    python indexing.py
    ```
    *Wait for "✅ Индексация завершена".*

---

### Step 4: Run the Bot (Docker)

Now that the database is ready, launch the full application.

```bash
docker-compose up --build