"""Read embedding model catalogs; never substitute a guessed model on failure."""

import json
import urllib.error
import urllib.request
from pathlib import Path
from collections.abc import Mapping

from rag.detect import API_ENDPOINTS, API_KEY_ENV, EMBEDDING_NAME

PROVIDERS = {
    name: (API_ENDPOINTS[name.lower()][0], API_KEY_ENV[name.lower()])
    for name in ("OpenAI", "Gemini", "OpenRouter")
}


class ApiKeyRejected(ValueError):
    """The catalog cannot be accessed with the supplied credentials."""


def api_models(provider: str, base_url: str, key: str) -> list[str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if provider == "Gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        headers = {"x-goog-api-key": key}
    else:
        url = base_url.rstrip("/") + ("/embeddings/models" if provider == "OpenRouter" else "/models")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=12) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise ApiKeyRejected("The provider rejected this API key.") from error
        raise ValueError(f"The model catalog returned HTTP {error.code}. Try again or choose another source.") from error
    except (OSError, ValueError) as error:
        raise ValueError("Could not load the model catalog. Check the connection or choose another source.") from error
    if not isinstance(payload, dict):
        raise ValueError("The provider returned an invalid model catalog.")
    if provider == "Gemini":
        models = [item["name"].removeprefix("models/") for item in payload.get("models", [])
                  if isinstance(item, dict) and isinstance(item.get("name"), str)
                  and "embedContent" in item.get("supportedGenerationMethods", [])]
    else:
        models = [item["id"] for item in payload.get("data", [])
                  if isinstance(item, dict) and isinstance(item.get("id"), str)
                  and (provider == "OpenRouter" or EMBEDDING_NAME.search(item["id"])
                       or "embeddings" in (item.get("architecture") or {}).get("output_modalities", []))]
    return sorted(set(models), key=str.casefold)


def environment_key(provider: str, environ: Mapping[str, str]) -> str:
    return next((environ[name] for name in PROVIDERS[provider][1] if environ.get(name)), "")


def downloads() -> list[dict]:
    from fastembed import TextEmbedding

    return sorted(TextEmbedding.list_supported_models(), key=lambda model: model["model"].casefold())


def cached_cpu_models(environ: Mapping[str, str]) -> list[str]:
    from huggingface_hub import try_to_load_from_cache
    from rag.embeddings import _fastembed_cache_dir

    cache = _fastembed_cache_dir(environ)
    if not Path(cache).is_dir():
        return []
    found = []
    for model in downloads():
        repo = (model.get("sources") or {}).get("hf")
        if not isinstance(repo, str):
            continue
        required = [model["model_file"], "tokenizer.json", "config.json", *model.get("additional_files", [])]
        paths = [try_to_load_from_cache(repo, name, cache_dir=cache) for name in required]
        if all(isinstance(path, str) and Path(path).is_file() for path in paths):
            found.append(model["model"])
    return sorted(set(found), key=str.casefold)
