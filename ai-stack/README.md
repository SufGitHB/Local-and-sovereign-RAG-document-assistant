# ai-stack

Local AI portal: **Open WebUI** as the chat interface, **LiteLLM** as the single
gateway to the models, **native Ollama** to run them on the host (CPU), and
**Keycloak** for user login (OpenID Connect). Every model is local: no prompt
or document leaves the machine.

The RAG assistant (`../pfe-rag`) is a separate compose project that joins the
same `ai-net` network.

## Architecture

```
  browser ──▶ open-webui :3000 ──▶ litellm :4000 ──▶ ollama (native) :11434
                  │    ▲               ▲                 qwen2.5:3b, bge-m3
       login      │    │ token         │
                  ▼    │               │
             keycloak :8081         rag-api :8001 (pfe-rag)
             realm "pfe"
```

| Component | Runs as | Config |
|---|---|---|
| `open-webui` | Docker (`ai-net`) | env vars in `docker-compose.yml`; settings are also stored in its database (`open-webui` volume) |
| `litellm` + `litellm-db` | Docker (`ai-net`) | `litellm/config.yaml` (model list), PostgreSQL for keys and the admin UI |
| `keycloak` | Docker (`ai-net`) | `keycloak/realm-pfe.json` (realm, `open-webui` client, `demo` user), imported on first start |
| `ollama` | native systemd service | `/etc/systemd/system/ollama.service.d/override.conf` (`OLLAMA_HOST=0.0.0.0:11434`) |
| secrets | — | `.env` (mode `600`, never committed — see `.env.example`) |

### Models exposed by LiteLLM

| Name | Ollama model | Use |
|---|---|---|
| `rag-chat-local` | `qwen2.5:3b` | chat / answer generation |
| `bge-m3` | `bge-m3` | embeddings only (not a chat model) |

## Login with Keycloak

Open WebUI shows a **Keycloak** button. Flow: Open WebUI → Keycloak login page
(`localhost:8081`) → back to Open WebUI with an OIDC token. Open WebUI stores no
password for these users.

- Keycloak admin console: http://localhost:8081 (user `admin`, password `KC_ADMIN_PASSWORD` in `.env`)
- Demo account in realm `pfe`: `demo` (password `KC_DEMO_PASSWORD` in `.env`)
- A new SSO account starts as **pending** in Open WebUI: approve it once in
  Admin Panel → Users.
- Two URLs on purpose: the browser uses `localhost:8081`, while open-webui
  talks to `keycloak:8080` inside the network (`KC_HOSTNAME` +
  `KC_HOSTNAME_BACKCHANNEL_DYNAMIC`).
- `start-dev` (HTTP, embedded database) is for this local prototype only;
  production would use `start` with TLS and PostgreSQL.

## Setup

```bash
cd ~/PFE-2026/ai-stack
cp .env.example .env && chmod 600 .env   # then fill in random values, e.g. openssl rand -hex 24
docker compose up -d
```

## Operating the stack

```bash
docker compose up -d                 # start
docker compose down                  # stop (data persists in volumes)
docker compose restart litellm       # after editing litellm/config.yaml
docker compose logs -f open-webui    # tail logs

systemctl status ollama              # native Ollama
ollama list                          # models pulled
ollama pull qwen2.5:3b               # CPU-friendly chat model (~2 GB)
```

## Access

- Open WebUI: http://localhost:3000
- Keycloak: http://localhost:8081
- LiteLLM API (OpenAI-compatible): http://localhost:4000/v1 (`Authorization: Bearer <LITELLM_MASTER_KEY>`)
- Ollama API: http://localhost:11434

## Notes

- Open WebUI keeps connection and OAuth settings in its database: once saved,
  editing env vars alone may not change them — use Admin Settings in the UI.
- `host.docker.internal` needs the `extra_hosts: host-gateway` mapping on Linux.
- `OLLAMA_HOST=0.0.0.0` opens the Ollama API to the LAN, not only to Docker.
- Keycloak sets secure cookies: browsers accept them on `http://localhost`,
  but use HTTPS for any other host name.
