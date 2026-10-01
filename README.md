# Assistant documentaire RAG local et souverain

Projet de fin d’études — ENSA Fès, filière ISDIA — réalisé en binôme par
**Soufiane EL AZZOUZI** et **Achraf BOUCHIKHI**.

An assistant that answers questions about internal documents, with sources,
where every model runs locally: no document or question leaves the machine.

| Folder | Content |
|---|---|
| [`ai-stack/`](ai-stack/README.md) | The portal: Open WebUI, LiteLLM, Keycloak (SSO), PostgreSQL — Ollama runs natively on the host |
| [`pfe-rag/`](pfe-rag/) | The RAG layer: FastAPI API, Qdrant, indexing, routing, evaluation scripts |

## Quick start

```bash
# 1. Portal
cd ai-stack && cp .env.example .env && chmod 600 .env   # fill in the values
docker compose up -d

# 2. RAG layer
cd ../pfe-rag && cp .env.example .env && chmod 600 .env  # same LITELLM_MASTER_KEY
mkdir -p docs && cp /path/to/your/*.pdf docs/            # documents to index (not versioned)
docker compose up -d --build
docker exec rag-api python -m rag.ingest --docs docs --reset

# 3. Evaluation
docker exec rag-api python -m eval.calibrate_threshold
docker exec rag-api python -m eval.evaluate_retrieval
```

Then open http://localhost:3000, log in with **Keycloak**, and pick the
`pfe-rag` model.

## CI/CD

`.gitlab-ci.yml` runs on every push: config checks (including "every LiteLLM
model is a local Ollama model"), the data-sovereignty test, and — on the
default branch — the build of the `rag-api` image, pushed to the GitLab
container registry. Deployment stays manual (the stack runs on a local machine).
