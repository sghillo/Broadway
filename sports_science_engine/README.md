# Sports Science & Vision AI Engine (`ssve`)

This package implements the **Agent Execution Manifest v1.0** (`config/manifest.yaml`). Each agent role in the manifest has its own sub-package. The orchestrator connects them in a weekly autonomous run: **ingest → index → synthesise → deliver**. A separate vision pipeline compares movements.

| Agent (manifest role) | Package | What it does |
|---|---|---|
| Data Mining & Ingestion Engineer | `ssve.ingestion` | PubMed E-utilities (esearch/efetch, DOI + abstract + date parsing), RSS/Atom regulatory feeds, JSON/CSV competition data. Requests are rate-limited and retried. Duplicates are removed by DOI. |
| RAG & Vector DB Architect | `ssve.rag` | Qdrant store (cosine, payload indexes on `kind`/`doi`, idempotent UUIDv5 point IDs). SentenceTransformers embeddings (default `BAAI/bge-small-en-v1.5`). vLLM client (OpenAI-compatible). **Strict DOI citation guardrails** (see below). |
| Content Synthesis Specialist | `ssve.synthesis` | Weekly executive digest (Snapshot, Evidence Briefs, Literature table, Regulatory Watch, Competition Highlights) and competitor memos. Sends Markdown + HTML email by SMTP, with **dry-run by default** (writes `.eml` files). |
| Computer Vision & Pose Engineer | `ssve.vision` | Ultralytics YOLO-pose 2D keypoints (COCO-17) with single-subject tracking, gap interpolation for low-confidence joints, multi-camera **DLT triangulation to 3D**, normalisation, joint angles, Savitzky–Golay smoothing, banded **DTW**, and per-joint deviation reports. |

## DOI citation guardrails

Every generated answer goes through `ssve.rag.guardrails.enforce`:

1. Only documents **with a DOI** are retrieved into the context window.
2. Each factual sentence must cite `[doi:…]`. Sentences without a citation are dropped.
3. A cited DOI must be one that was **retrieved for this question**. If it wasn't (a hallucinated or "remembered" reference), the sentence is dropped and the DOI is logged in `rejected_citations`.
4. If nothing survives, the answer becomes an explicit *"Insufficient peer-reviewed evidence…"* message.

## Quick start

```bash
cd sports_science_engine
pip install -e ".[all,dev]"        # or just `pip install -e .[dev]` for the offline core
pytest                             # 26 tests, fully offline

# Production services
docker run -p 6333:6333 qdrant/qdrant
vllm serve meta-llama/Llama-3.1-8B-Instruct      # OpenAI-compatible on :8000

ssve weekly --config config/manifest.yaml        # writes reports/digest_*.md, memo_*.md
ssve ask "Does sleep extension improve sprint performance?"
ssve compare coach_demo.mp4 athlete_rep.mp4 --dims 2d   # or two .npy keypoint arrays
```

Environment variables: `VLLM_BASE_URL`, `VLLM_MODEL`, `VLLM_API_KEY`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`.

### Offline / degraded mode
Heavy dependencies are optional. If `sentence-transformers` or `qdrant-client` is missing, the orchestrator falls back to a hashing embedder and an in-memory store. Set `rag.llm: extractive` to run without a model server. The guardrails apply in every mode.

## Keypoint array conventions

- 2D: `(T, 17, 3)` → `x, y, confidence` (YOLO output). Pass `has_conf=True`.
- 3D: `(T, 17, 3)` → `X, Y, Z` (from `triangulate_sequence`). Pass `has_conf=False`.

## Compliance notes
- Check each source's terms of service and `robots.txt` before adding scraping targets. Competition data providers often license their feeds.
- Athlete video and keypoints are personal (often biometric) data. Under UK/EU GDPR, record a lawful basis, keep footage only as long as needed, and prefer storing keypoints over raw video.
