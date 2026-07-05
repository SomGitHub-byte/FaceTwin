# FaceTwin 🎭

**An AI agent that finds your celebrity look-alikes — built for the Kaggle x Google Vibe Coding Capstone.**

Upload a photo, and a multi-agent pipeline analyzes your facial features, compares them against a database of public figures using face-embedding similarity, and returns your top 5 matches with a percentage score and a short AI-generated explanation of the resemblance.

> ⚠️ **For entertainment only.** FaceTwin is based on facial feature similarity, not personality, genetics, or any scientific trait analysis.

---

## How it works

1. You upload a photo.
2. An agent pipeline detects your face and extracts a numeric embedding of your facial features.
3. That embedding is compared against a precomputed database of celebrity face embeddings.
4. The top 5 closest matches are ranked and converted into similarity percentages.
5. An LLM writes a short, tasteful explanation for each match (e.g. *"82% match with Elon Musk — similar jaw structure and brow shape"*).

---

<img width="1763" height="1340" alt="image" src="https://github.com/user-attachments/assets/ea301e19-6d92-48ab-84c5-c153eb6ed98f" />


## Architecture

FaceTwin is built as a **multi-agent system** on **Google's Agent Development Kit (ADK)**, with all face-processing logic exposed through a standalone **MCP server** rather than imported directly into the agents.

```
orchestrator_agent (root LlmAgent)
├── intake_agent       → validates the upload, runs security checks, detects the face
├── embedding_agent     → generates the face embedding, queries the celebrity database
├── ranking_agent       → scores and ranks the top 5 matches
└── narrator_agent      → generates the explanation for each match via Gemini
```

The agents never touch image-processing code directly — every face operation (detection, embedding, database lookup, moderation) is exposed as a tool on a separate MCP server and called via `McpToolset`. This keeps the reasoning layer (agents) cleanly separated from the processing layer (tools).

```
┌─────────────┐      MCP       ┌──────────────────┐
│  ADK Agents │ ─────────────► │    MCP Server     │
│ (reasoning) │ ◄───────────── │  (face tools)      │
└─────────────┘                └──────────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agents | Google ADK (`google-adk`) |
| LLM | Gemini (`google-genai`) |
| Tool server | MCP (Python MCP SDK / FastMCP) |
| Face detection | MediaPipe |
| Face embeddings | `deepface` (Facenet512 / ArcFace) |
| Similarity DB | Precomputed embeddings, stored offline (`.pkl`) |
| Frontend | Gradio |
| Deployment | Docker + Google Cloud Run |

---

## Repository structure

```
facetwin/
├── agents/
│   ├── orchestrator.py
│   ├── intake_agent.py
│   ├── embedding_agent.py
│   ├── ranking_agent.py
│   └── narrator_agent.py
├── mcp_server/
│   ├── server.py
│   └── tools/
│       ├── face_detection.py
│       ├── embeddings.py
│       ├── celebrity_db.py
│       └── moderation.py
├── scripts/
│   └── build_celebrity_db.py
├── data/
│   └── celebrity_embeddings.pkl
├── app.py
├── Dockerfile
├── deploy.sh
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Security

- **No persistent storage** — uploaded images are processed in memory and discarded immediately after the embedding is generated. Nothing is written to disk or a database.
- **Input validation** — enforced file type (JPG/PNG), max file size, and image dimension checks before any processing.
- **Content moderation gate** — images are screened before reaching the embedding step; non-human or inappropriate uploads are rejected.
- **Secrets management** — API keys are loaded from environment variables via `.env` (git-ignored). See `.env.example` for required variables.
- **Rate limiting** — basic per-session throttling on the frontend to prevent API abuse.
- **Least-privilege tools** — the MCP server only exposes the four tools required by the agents; nothing else is reachable.

---

## Getting started

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### Setup

```bash
git clone https://github.com/<your-username>/facetwin.git
cd facetwin

# Install dependencies
uv sync

# Configure environment variables
cp .env.example .env
# then add your GEMINI_API_KEY to .env
```

### Build the celebrity database (one-time)

```bash
uv run scripts/build_celebrity_db.py
```

### Run locally

```bash
uv run app.py
```

The Gradio app will start on `http://localhost:7860`.

### Run the MCP server standalone (for testing/debugging)

```bash
uv run mcp_server/server.py
```

---

## Deployment

FaceTwin is containerized and deployable to Google Cloud Run in one command.

```bash
./deploy.sh
```

This builds the Docker image and deploys it to Cloud Run, injecting configuration via environment variables / Secret Manager — no credentials are baked into the image. See `deploy.sh` for the exact `gcloud` commands used.

---

## Privacy

FaceTwin does not store, log, or retain uploaded photos. Images exist only in memory for the duration of a single request and are discarded once the face embedding is generated.

---

## Hackathon context

Built for the **Kaggle x Google Vibe Coding Capstone**, demonstrating:

- A multi-agent system built with Google ADK
- A standalone MCP server for tool orchestration
- Practical security practices for a user-facing AI app handling biometric data
- A fully deployable, containerized service on Google Cloud Run

---

## Disclaimer

FaceTwin is a novelty application for entertainment purposes. It does not perform, claim, or imply any scientific, medical, psychological, or genetic analysis. Facial similarity does not reflect personality, character, or any other trait.
