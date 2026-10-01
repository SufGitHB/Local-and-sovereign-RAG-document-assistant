"""
Data-sovereignty startup guard.

Resolves LLM_MODEL and EMBED_MODEL through LiteLLM's /model/info endpoint
and refuses to start the API if either one maps to a non-local provider.
This is what makes the guarantee real: config.py can say "rag-chat-local",
but only the gateway's own model_list decides where a call actually goes
(see ~/PFE-2026/ai-stack/litellm/config.yaml) — so we ask the gateway, not
trust the name.
"""

import requests

from . import config

# Providers considered local for this project. Anything else means a call
# leaves the host (e.g. "openrouter").
LOCAL_PROVIDERS = {"ollama"}


def _resolve_provider(model_name: str, model_info: list[dict]) -> str:
    """Return the provider prefix (e.g. "ollama") LiteLLM would actually
    dispatch model_name to, by reading litellm_params.model — a string
    shaped like "<provider>/<model>"."""
    for entry in model_info:
        if entry.get("model_name") == model_name:
            litellm_model = entry["litellm_params"]["model"]
            return litellm_model.split("/", 1)[0]
    raise RuntimeError(
        f"Model '{model_name}' is not defined in LiteLLM (checked /model/info) — "
        f"check litellm/config.yaml"
    )


def assert_local_models() -> None:
    """Fail loudly at startup if LLM_MODEL or EMBED_MODEL resolve to a
    non-local provider. Logs the resolved provider for both either way."""
    resp = requests.get(
        f"{config.LITELLM_URL}/model/info",
        headers={"Authorization": f"Bearer {config.LITELLM_API_KEY}"},
        timeout=10,
    )
    resp.raise_for_status()
    model_info = resp.json()["data"]

    for model_name in (config.LLM_MODEL, config.EMBED_MODEL):
        provider = _resolve_provider(model_name, model_info)
        print(f"[startup] {model_name} -> provider={provider}", flush=True)
        if provider not in LOCAL_PROVIDERS:
            raise RuntimeError(
                f"Data sovereignty violation: '{model_name}' resolves to provider "
                f"'{provider}' (not local). Refusing to start."
            )
