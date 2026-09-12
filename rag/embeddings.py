"""Embedding providers behind one small interface: embed_documents / embed_query.

`build_embedder` turns the user's `rag setup` choice into an object the vector
backends can call. Heavy imports (fastembed, openai, sentence-transformers) stay
inside the provider that needs them, so the light install never loads PyTorch.
"""

from __future__ import annotations

import os
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FULL_EXTRA_HINT = (
    "Install the full extra (uv tool install 'ragmcp[full]') or pick another option in `rag setup`."
)


def _fastembed_cache_dir(environ: Mapping[str, str] | None = None) -> str:
    environ = os.environ if environ is None else environ
    return environ.get("FASTEMBED_CACHE_PATH") or os.path.join(
        environ.get("XDG_CACHE_HOME") or os.path.join(str(Path.home()), ".cache"), "fastembed"
    )


class _hf_online:
    """The MCP server pins Hugging Face to offline mode for the PyTorch stack; a small
    ONNX model still has to download once. Lift the pin only while it loads."""

    def __enter__(self):
        self.saved = {name: os.environ.pop(name, None) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        try:
            import huggingface_hub.constants as constants

            self.constants, self.was = constants, constants.HF_HUB_OFFLINE
            constants.HF_HUB_OFFLINE = False
        except Exception:
            self.constants = None
        return self

    def __exit__(self, *_):
        for name, value in self.saved.items():
            if value is not None:
                os.environ[name] = value
        if self.constants is not None:
            self.constants.HF_HUB_OFFLINE = self.was


class FastEmbedWrapper:
    """Small ONNX models via fastembed: CPU-only, downloaded once, no PyTorch."""

    def __init__(self, model_name: str):
        with _hf_online():
            try:
                from fastembed import TextEmbedding
            except ImportError as error:
                raise RuntimeError(
                    "The local embedding model needs the fastembed package. Reinstall ragmcp or run `rag setup`."
                ) from error
            self.model_name = model_name
            self.model = TextEmbedding(model_name=model_name, cache_dir=_fastembed_cache_dir())

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.embed(texts, batch_size=64)]

    def embed_query(self, text):
        return next(iter(self.model.query_embed(text))).tolist()


class DirectOllamaGPUWrapper:
    """Send true vector batches directly to Ollama's /api/embed."""

    def __init__(self, model, base_url):
        import requests

        self._requests = requests
        self.model = model
        self.model_name = model
        self.base_url = base_url

    def _verify_gpu(self):
        try:
            resp = self._requests.get(f"{self.base_url}/api/ps", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for m in models:
                    if self.model in m.get("name", ""):
                        proc = m.get("details", {}).get("processor", "") or m.get("processor", "")
                        if "CPU" in str(proc) and "100% CPU" in str(proc):
                            raise RuntimeError(f"OLLAMA GPU FAILURE: {self.model} is running on {proc}! Aborting.")
        except Exception as e:
            if "OLLAMA GPU FAILURE" in str(e):
                raise

    def embed_documents(self, texts):
        if not texts:
            return []
        # Process in GPU batches of 512
        batch_size = 512
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=180,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama embedding failed ({resp.status_code}): {resp.text}")
            all_embeddings.extend(resp.json().get("embeddings", []))
        self._verify_gpu()
        return all_embeddings

    def embed_query(self, text):
        resp = self._requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": [text]},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama query embedding failed ({resp.status_code}): {resp.text}")
        embeddings = resp.json().get("embeddings", [])
        self._verify_gpu()
        return embeddings[0] if embeddings else []


class OpenAICompatibleWrapper:
    """One client for any provider that speaks the OpenAI embeddings API shape:
    real OpenAI, Gemini's and OpenRouter's OpenAI-compatible modes, Ollama's /v1.
    This is the one provider option where text leaves the machine."""

    # The provider supports batches and the SDK retries transient errors.
    # Repeating a failed batch as individual queries multiplies requests/cost.
    retry_batch_as_queries = False

    def __init__(self, model_name, api_key=None, base_url=None):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("The API embedding option needs the openai package. Reinstall ragmcp.") from error
        # Fail fast on a wrong URL or key: the setup probe should answer in seconds.
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=1, timeout=30)
        self.model_name = model_name

    def _create_with_retry(self, input_val):
        # The SDK already classifies retryable errors and owns the retry budget.
        return self.client.embeddings.create(model=self.model_name, input=input_val)

    def embed_documents(self, texts):
        if not texts:
            return []
        resp = self._create_with_retry(texts)
        items = sorted(resp.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(texts))):
            raise ValueError("Invalid embedding response: missing or duplicate input indices.")
        vectors = [item.embedding for item in items]
        dimensions = len(vectors[0])
        if not dimensions or any(
            len(vector) != dimensions or not all(math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ValueError("Invalid embedding response: empty, inconsistent or non-finite vectors.")
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class STWrapper:
    """sentence-transformers on the selected device. Needs the [full] extra."""

    def __init__(self, model_name, device):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(f"sentence-transformers is not installed. {FULL_EXTRA_HINT}") from error
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


class FastEmbedReranker:
    """ONNX cross-encoder via fastembed: reranking without PyTorch."""

    def __init__(self, model_name: str):
        with _hf_online():
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as error:
                raise RuntimeError("The local reranker needs the fastembed package. Reinstall ragmcp.") from error
            self.model_name = model_name
            self.model = TextCrossEncoder(model_name=model_name, cache_dir=_fastembed_cache_dir())

    def predict(self, pairs):
        """Same shape as sentence-transformers' CrossEncoder.predict: one score per (query, passage)."""
        if not pairs:
            return []
        query = pairs[0][0]
        return list(self.model.rerank(query, [passage for _, passage in pairs]))


def api_key_from_environment(base_url: str | None) -> str | None:
    """Keys never have to be copied into the config file: match the endpoint to its usual variable."""
    host = (base_url or "").lower()
    names = (
        ("OPENROUTER_API_KEY",) if "openrouter" in host
        else ("GEMINI_API_KEY", "GOOGLE_API_KEY") if "googleapis" in host
        else ("OPENAI_API_KEY",)
    )
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def build_embedder(config: Mapping[str, Any], device: str = "cpu"):
    """Return the embedder for this config, or None when only BM25 should run."""
    provider = str(config.get("embed_provider") or "none").lower()
    model_name = str(config.get("embed_model") or "")
    if provider == "none":
        return None
    if provider == "fastembed":
        return FastEmbedWrapper(model_name or "BAAI/bge-small-en-v1.5")
    if provider == "ollama":
        return DirectOllamaGPUWrapper(model_name or "nomic-embed-text", config.get("ollama_url") or "http://localhost:11434")
    if provider == "openai_compatible":
        base_url = config.get("embed_base_url") or None
        return OpenAICompatibleWrapper(
            model_name or "text-embedding-3-small",
            api_key=config.get("embed_api_key") or api_key_from_environment(base_url),
            base_url=base_url,
        )
    if provider == "sentencetransformer":
        return STWrapper(model_name or "all-MiniLM-L6-v2", device)
    raise ValueError(f"Unknown embedding provider: {provider}. Run `rag setup` to choose a supported provider.")


def probe_dimensions(embedder) -> int:
    """Embed one sentence to learn the true vector size of any provider/model."""
    vector = embedder.embed_query("dimension probe")
    if not vector:
        raise RuntimeError(f"{embedder.model_name} returned an empty embedding.")
    return len(vector)
