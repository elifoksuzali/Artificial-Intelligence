# GPT

Large-language-model work: prompting experiments, generative-AI demos, and a multi-day **travel-agent chatbot** built with Retrieval-Augmented Generation (RAG) over Turkish tour data. Integrations cover **OpenAI, Gemini, Ollama** and **Whisper**, with **Qdrant** as the vector store.

## Top-level notebooks

| File | Description |
|------|-------------|
| `Prompting.ipynb` | Prompt-engineering experiments |
| `Api_Promt_Translater.ipynb` | Prompt/translation via API |
| `OpenSource_GenAI.ipynb`, `OpenSource-Models-x-Colab-GenAI.ipynb` | Open-source generative models on Colab |
| `WebScraping.ipynb` | Data collection via web scraping |
| `whisper-proje.ipynb` | Speech-to-text with Whisper |

## `GPT_Agent/` — travel chatbot, day by day

A progressive build of a tour/travel assistant. Each day iterates on the architecture.

### `1_day/` — Gemini chatbot + RAG foundation
Image/audio/PDF analysis bot with a modular layout: `agent/`, `extract/`, `database/` (Qdrant), `prompting/`, `main/`. Tour documents are cleaned (`clean data/`) and indexed into Qdrant. Has its own `README.md` and `requirements.txt`.

### `3_day/` — provider integrations (PHP + Python)
- `gemini/` — `gemini.php`, `geminiTour.py`
- `openai/` — `index.php`, `tripadvisor.php`, prompt notes

### `4_day/` — agent service ("agent-bot")
A more complete agent: `agent.py`, `api.py`, `context.py`, `mcp.py`, `rerank.py`, `thread.py`, `database/db.py`, `qdrantDb.py`, plus `requirements.txt`. Includes reranking and an MCP-style tool layer.

### `5_day/` — Ollama integration + Qdrant hybrid search
`ollama_entegretion/` implements **hybrid search (semantic + keyword)** over tour data in Qdrant, optimized for Turkish embeddings. Structured into `analysis/`, `embed/`, `handle/`, `prompt/`, `pydict/`, `qdrantDB/`. See its `README.md` for setup, data schema and query examples. Sample data in `sata/` (`konaklama_data.csv`, `ulasim_data.csv`, `birlesik.json`).

## Stack

OpenAI · Google Gemini · Ollama · OpenAI Whisper · Qdrant · sentence-transformers · Flask/PHP · Google Colab

## Running the agents

Each `GPT_Agent/*/` sub-project is standalone:

```bash
pip install -r requirements.txt     # inside the chosen day's folder
# start Qdrant (Docker):
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

Then run that day's entry point (e.g. `main.py` / `api.py`). Provide API keys via environment or a local `.env` (not committed).
