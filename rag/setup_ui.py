"""Interactive selection of an embedding provider and its real model list.

The first question chooses where embeddings run.  Model names are shown only
after that provider has been selected, and API names come from the provider's
catalog instead of a hand-written list.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlparse

from rag import detect, model_catalog, prompts, settings


API_PROVIDER_ORDER = ("OpenRouter", "OpenAI", "Gemini")
API_DISPLAY_NAMES = {"Gemini": "Google Gemini"}


def _url(message, *, stdin, stdout, default=""):
    while True:
        value = prompts.text(message, default=default, stdin=stdin, stdout=stdout).strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            return value
        stdout.write("Enter a full URL beginning with http:// or https://.\n")


def _api_providers() -> list[str]:
    return [name for name in API_PROVIDER_ORDER if name in model_catalog.PROVIDERS]


def _api_name(provider: str) -> str:
    return API_DISPLAY_NAMES.get(provider, provider)


def _api_key_available(provider: str, saved_provider: str | None, saved_key: str, environ: Mapping[str, str]) -> bool:
    return bool(model_catalog.environment_key(provider, environ) or (provider == saved_provider and saved_key))


def _api(*, stdin, stdout, environ, provider: str | None = None):
    """Choose a provider, load its live embedding catalog, then choose a model.

    ``provider`` is used by the provider-first screen.  Leaving it unset keeps
    the function useful to callers that want the provider sub-question.
    """
    # The direct provider-first terminal path uses the readable order above;
    # the numbered stdin compatibility path keeps the historical API order.
    providers = list(model_catalog.PROVIDERS) if provider is None else _api_providers()
    saved = settings.embedding_config(environ)
    saved_base = saved["embed_base_url"].rstrip("/") if saved["embed_provider"] == "openai_compatible" else ""
    saved_provider = next((name for name in providers if model_catalog.PROVIDERS[name][0].rstrip("/") == saved_base), None)
    saved_endpoint = bool(saved_base and saved_provider is None)
    fixed_provider = provider if provider in providers or provider == "Other" else None

    if fixed_provider is None:
        choice = providers.index(saved_provider) if saved_provider else len(providers) if saved_endpoint else next(
            (index for index, name in enumerate(providers) if _api_key_available(name, saved_provider, saved["embed_api_key"], environ)), 0)
    else:
        choice = 0

    while True:
        try:
            if fixed_provider is None:
                labels = [
                    f"{_api_name(name)} · key available · hosted embeddings" if _api_key_available(name, saved_provider, saved["embed_api_key"], environ)
                    else f"{_api_name(name)} · key needed · hosted embeddings"
                    for name in providers
                ]
                if saved_endpoint:
                    labels.append(f"Saved endpoint · {saved_base}")
                choice = prompts.select(
                    "Which provider should supply embeddings?",
                    [*labels, "Another OpenAI-compatible endpoint…"],
                    default=choice,
                    stdin=stdin,
                    stdout=stdout,
                )
                if choice < len(providers):
                    selected_provider = providers[choice]
                    base = model_catalog.PROVIDERS[selected_provider][0]
                    key = model_catalog.environment_key(selected_provider, environ)
                    saved_key = ""
                    if not key and selected_provider == saved_provider:
                        key = saved_key = saved["embed_api_key"]
                    if not key:
                        key = saved_key = prompts.text(
                            f"Paste your {_api_name(selected_provider)} API key",
                            secret=True,
                            stdin=stdin,
                            stdout=stdout,
                        ).strip()
                        if not key:
                            if not prompts.interactive(stdin, stdout):
                                raise ValueError("An API key is required.")
                            stdout.write("An API key is needed to load the provider's model list.\n")
                            continue
                elif saved_endpoint and choice == len(providers):
                    selected_provider, base = "Other", saved_base
                    key = saved_key = saved["embed_api_key"]
                else:
                    selected_provider = "Other"
                    base = _url(
                        "API base URL (include /v1, e.g. http://localhost:1234/v1)",
                        stdin=stdin,
                        stdout=stdout,
                    )
                    key = saved_key = prompts.text(
                        "API key (leave empty for a local server)",
                        secret=True,
                        stdin=stdin,
                        stdout=stdout,
                    ).strip()
            else:
                selected_provider = fixed_provider
                if selected_provider == "Other":
                    base = _url(
                        "API base URL (include /v1, e.g. http://localhost:1234/v1)",
                        stdin=stdin,
                        stdout=stdout,
                    )
                    key = saved_key = prompts.text(
                        "API key (leave empty for a local server)",
                        secret=True,
                        stdin=stdin,
                        stdout=stdout,
                    ).strip()
                else:
                    base = model_catalog.PROVIDERS[selected_provider][0]
                    key = model_catalog.environment_key(selected_provider, environ)
                    saved_key = ""
                    if not key and selected_provider == saved_provider:
                        key = saved_key = saved["embed_api_key"]
                    if not key:
                        key = saved_key = prompts.text(
                            f"Paste your {_api_name(selected_provider)} API key",
                            secret=True,
                            stdin=stdin,
                            stdout=stdout,
                        ).strip()
                        if not key:
                            if not prompts.interactive(stdin, stdout):
                                raise ValueError("An API key is required.")
                            stdout.write("An API key is needed to load the provider's model list.\n")
                            raise prompts.Cancelled

            models = None
            while True:
                stdout.write(f"Loading {_api_name(selected_provider)} embedding models…\n")
                stdout.flush()
                try:
                    models = model_catalog.api_models(selected_provider, base, key)
                    break
                except model_catalog.ApiKeyRejected as error:
                    if not prompts.interactive(stdin, stdout):
                        raise
                    stdout.write(f"{error}\n")
                    action = prompts.select(
                        "How would you like to continue?",
                        ["Replace API key", "Back to providers"],
                        stdin=stdin,
                        stdout=stdout,
                    )
                    if action == 1:
                        if fixed_provider is not None:
                            raise prompts.Cancelled
                        break
                    key = saved_key = prompts.text(
                        "Paste a replacement API key",
                        secret=True,
                        stdin=stdin,
                        stdout=stdout,
                        validate=lambda value: True if value.strip() else "Enter an API key, or press Esc to go back.",
                    ).strip()
                except ValueError as error:
                    stdout.write(f"{error}\n")
                    if not prompts.interactive(stdin, stdout):
                        raise
                    if selected_provider == "Other":
                        models = []
                        break
                    action = prompts.select(
                        "Could not load this provider's model list. What would you like to do?",
                        ["Try again", "Back to sources"],
                        stdin=stdin,
                        stdout=stdout,
                    )
                    if action == 1:
                        raise prompts.Cancelled
                    continue
            if models is None:
                continue
            if not models:
                stdout.write("No embedding models were returned by this provider.\n")
            back_label = "Back" if fixed_provider is not None else "Back to providers"
            options = [*models, back_label]
            if selected_provider == "Other":
                options.append("Enter an exact model ID…")
            current_model = saved["embed_model"] if base.rstrip("/") == saved_base else ""
            stdout.write("These model IDs came from the provider just selected; choose one to continue.\n")
            selected = prompts.select(
                f"Which {_api_name(selected_provider)} embedding model should RAG use?",
                options,
                default=models.index(current_model) if current_model in models else 0,
                stdin=stdin,
                stdout=stdout,
            )
            if selected == len(models):
                if fixed_provider is not None:
                    raise prompts.Cancelled
                continue
            if selected < len(models):
                model = models[selected]
            else:
                model = prompts.text(
                    "Model ID (copy the exact ID from your server's /models response)",
                    stdin=stdin,
                    stdout=stdout,
                ).strip()
                if not model:
                    continue
            return {
                "embed_provider": "openai_compatible",
                "embed_model": model,
                "embed_base_url": base,
                "embed_api_key": saved_key,
            }
        except prompts.Cancelled:
            # Esc always returns to the provider/source screen. Reopening the
            # same prompt made it look as if the key or model question was
            # stuck.
            raise


def _ollama(*, detection, current, stdin, stdout, environ):
    info = detection.get("ollama") or {}
    url = info.get("url") or current["ollama_url"] or settings.DEFAULTS["ollama_url"]
    models = [model for model in info.get("embedding_models") or [] if isinstance(model, str) and model]
    if not info.get("running"):
        stdout.write(f"Ollama is not reachable at {url}. Start Ollama, then choose this option again.\n")
        # When detection already supplied API-key state, the provider rows are
        # available immediately on the source screen. With no usable local
        # runtime, move straight to that provider question instead of asking
        # the user to select the same recovery action twice.
        if prompts.interactive(stdin, stdout) and not detection.get("api_keys"):
            stdout.write("Choose a hosted provider instead.\n")
            return _api(stdin=stdin, stdout=stdout, environ=environ)
        raise prompts.Cancelled
    if not models:
        stdout.write("Ollama is running, but it has no embedding model. Pull one with `ollama pull <model>` first.\n")
        raise prompts.Cancelled
    default = models.index(current["embed_model"]) if current["embed_provider"] == "ollama" and current["embed_model"] in models else 0
    selected = prompts.select(
        "Which installed Ollama embedding model should RAG use?",
        [*models, "Back"],
        default=default,
        stdin=stdin,
        stdout=stdout,
    )
    if selected == len(models):
        raise prompts.Cancelled
    return {"embed_provider": "ollama", "embed_model": models[selected], "ollama_url": url}


def _size_label(size_in_gb: object) -> str:
    try:
        size = float(size_in_gb)
    except (TypeError, ValueError):
        return "size unknown"
    if size < 1:
        return f"~{round(size * 1000):g} MB"
    return f"~{size:g} GB"


def _download(*, stdin, stdout):
    stdout.write("Loading the supported local CPU model catalog…\n")
    stdout.flush()
    models = model_catalog.downloads()
    if not models:
        stdout.write("No local CPU models are available from the installed runtime.\n")
        raise prompts.Cancelled
    labels = []
    for model in models:
        description = str(model.get("description") or "").split(",")
        language = next((part.strip() for part in description if any(word in part.lower() for word in ("english", "multilingual", "languages", "chinese", "german", "spanish"))), "")
        suffix = " · ".join(part for part in (language, _size_label(model.get("size_in_GB"))) if part)
        labels.append(f"{model['model']} · {suffix}" if suffix else str(model["model"]))
    selected = prompts.select(
        "Which supported model should RAG download for local CPU search?",
        [*labels, "Back"],
        stdin=stdin,
        stdout=stdout,
    )
    if selected == len(models):
        raise prompts.Cancelled
    return {"embed_provider": "fastembed", "embed_model": models[selected]["model"]}


def _sentence_transformer(*, detection, stdin, stdout):
    cached = [model for model in detection.get("cached_models") or [] if isinstance(model, dict) and model.get("path")]
    labels = [f"{model['label']} · already downloaded" for model in cached]
    labels.append("Enter a model path or repository ID…")
    labels.append("Back")
    selected = prompts.select(
        "Which local sentence-transformers model should RAG use?",
        labels,
        stdin=stdin,
        stdout=stdout,
    )
    if selected == len(cached) + 1:
        raise prompts.Cancelled
    if selected < len(cached):
        # Persist the repository ID, never this machine's cache snapshot path.
        # The transformer runtime resolves it from its normal cache directory.
        return {"embed_provider": "sentencetransformer", "embed_model": cached[selected]["label"]}
    model = prompts.text(
        "Model path or repository ID (owner/model)",
        stdin=stdin,
        stdout=stdout,
        validate=lambda value: True if value.strip() else "Enter a model path or repository ID, or press Esc to go back.",
    ).strip()
    if not model:
        raise prompts.Cancelled
    return {"embed_provider": "sentencetransformer", "embed_model": model}


def _other(*, detection, stdin, stdout, environ):
    selected = prompts.select(
        "Which additional local source do you want to use?",
        [
            "Ollama on another computer · enter its address",
            "OpenAI-compatible server · enter its address",
            "Local sentence-transformers model · choose a cached model or enter one",
            "Back",
        ],
        stdin=stdin,
        stdout=stdout,
    )
    if selected == 3:
        raise prompts.Cancelled
    if selected == 0:
        url = _url("Ollama server URL (e.g. http://192.168.1.10:11434)", stdin=stdin, stdout=stdout)
        stdout.write("Looking for embedding models at that Ollama server…\n")
        stdout.flush()
        result = detect.detect_ollama(url)
        models = result["embedding_models"]
        if not models:
            stdout.write("No embedding models found there. Check the address and pull an embedding model first.\n")
            raise prompts.Cancelled
        model = prompts.select(
            "Which embedding model should RAG use?",
            [*models, "Back"],
            stdin=stdin,
            stdout=stdout,
        )
        if model == len(models):
            raise prompts.Cancelled
        return {"embed_provider": "ollama", "embed_model": models[model], "ollama_url": url}
    if selected == 1:
        return _api(provider="Other", stdin=stdin, stdout=stdout, environ=environ)
    return _sentence_transformer(detection=detection, stdin=stdin, stdout=stdout)


def _source_choices(detection, current, environ):
    ollama = detection.get("ollama") or {}
    ollama_models = ollama.get("embedding_models") or []
    if ollama.get("running") and ollama_models:
        ollama_label = f"Ollama · {len(ollama_models)} installed · runs on this computer"
    elif ollama.get("running"):
        ollama_label = "Ollama · running, no embedding model installed"
    else:
        ollama_label = "Ollama · not running on this computer"

    api_labels = []
    saved = settings.embedding_config(environ)
    saved_base = saved["embed_base_url"].rstrip("/") if saved["embed_provider"] == "openai_compatible" else ""
    for provider in _api_providers():
        available = _api_key_available(
            provider,
            next((name for name in _api_providers() if model_catalog.PROVIDERS[name][0].rstrip("/") == saved_base), None),
            saved["embed_api_key"],
            environ,
        )
        api_labels.append(f"{_api_name(provider)} · {'key available' if available else 'key needed'} · hosted embeddings")

    cached_count = len(detection.get("cached_models") or [])
    return [
        ollama_label,
        "Local CPU · download a supported model once · no account",
        *api_labels,
        "Another Ollama/API server · enter its address",
        f"Custom local model · {cached_count} cached" if cached_count else "Custom local model · choose a path or repository ID",
        "BM25 only · exact words, no embedding model",
    ]


def _source_default(choices, detection, current, environ):
    provider = current["embed_provider"]
    if provider == "ollama":
        return 0
    if provider == "fastembed":
        return 1
    if provider == "openai_compatible":
        base = current["embed_base_url"].rstrip("/")
        for index, name in enumerate(_api_providers(), 2):
            if model_catalog.PROVIDERS[name][0].rstrip("/") == base:
                return index
        return 2 + len(_api_providers())
    if provider == "sentencetransformer":
        return 2 + len(_api_providers()) + 1
    if provider == "none":
        ollama = detection.get("ollama") or {}
        if ollama.get("running") and ollama.get("embedding_models"):
            return 0
        for index, name in enumerate(_api_providers(), 2):
            if _api_key_available(name, None, "", environ):
                return index
        return 1
    return 0


def _run_line_input(*, stdin, stdout, stderr, environ, verify):
    """Keep the numbered stdin contract for scripts and older automation.

    A real terminal uses the provider-first chooser below. Piped input has no
    cursor or back-navigation, so retaining its established numbered order is
    the least surprising compatibility path.
    """
    detection = detect.detect_all(environ)
    current = settings.embedding_config(environ)
    installed, labels = [], []
    ollama = detection.get("ollama") or {}
    for model in ollama.get("embedding_models") or []:
        installed.append({
            "embed_provider": "ollama",
            "embed_model": model,
            "ollama_url": ollama.get("url", settings.DEFAULTS["ollama_url"]),
        })
        labels.append(f"{model} · Ollama")
    for runtime in detection.get("local_runtimes") or []:
        for model in runtime.get("embedding_models") or []:
            installed.append({
                "embed_provider": "openai_compatible",
                "embed_model": model,
                "embed_base_url": runtime["base_url"],
                "embed_api_key": "local",
            })
            labels.append(f"{model} · {runtime['label']}")
    for model in detection.get("cached_models") or []:
        installed.append({"embed_provider": "sentencetransformer", "embed_model": model["label"]})
        labels.append(f"{model['label']} · local files")
    for model in detection.get("cached_cpu_models") or []:
        installed.append({"embed_provider": "fastembed", "embed_model": model})
        labels.append(f"{model} · local CPU")
    if not installed:
        stdout.write("No local embedding models found. Choose an API or download a model below.\n")
    choices = [*labels, "Choose an API model…", "Download a local CPU model…", "Use BM25 only · no model", "Other model source…"]
    selected = prompts.select(
        "Which embedding model would you like to use?",
        choices,
        default=0,
        stdin=stdin,
        stdout=stdout,
    )
    if selected < len(installed):
        choice = installed[selected]
    elif selected == len(installed):
        choice = _api(stdin=stdin, stdout=stdout, environ=environ)
    elif selected == len(installed) + 1:
        choice = _download(stdin=stdin, stdout=stdout)
    elif selected == len(installed) + 2:
        choice = {"embed_provider": "none", "embed_model": ""}
    else:
        choice = _other(detection=detection, stdin=stdin, stdout=stdout, environ=environ)
    config = dict(settings.DEFAULTS)
    config.update({key: value for key, value in choice.items() if key in settings.DEFAULTS})
    return settings.finish_setup(config, stdout=stdout, stderr=stderr, environ=environ, verify=verify)


def run(*, stdin, stdout, stderr, environ=None, verify=None):
    environ = os.environ if environ is None else environ
    if not prompts.interactive(stdin, stdout):
        try:
            return _run_line_input(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
        except prompts.Cancelled as error:
            raise ValueError("Setup ended before a complete embedding choice was made.") from error
    stdout.write("Exact-word search (BM25) is always on. Add an embedding model to match related wording.\n")
    stdout.write("Choose where it should run; the next screen shows only models from that source.\n\n")
    stdout.write("Retrieval only; Studio or a connected app chooses the answer model.\n\n")
    while True:
        stdout.flush()
        detection = detect.detect_all(environ)
        current = settings.embedding_config(environ)
        choices = _source_choices(detection, current, environ)
        default = _source_default(choices, detection, current, environ)
        selected = prompts.select(
            "Where should RAG run the embedding model?",
            choices,
            default=default,
            stdin=stdin,
            stdout=stdout,
        )
        try:
            api_start = 2
            if selected == 0:
                choice = _ollama(detection=detection, current=current, stdin=stdin, stdout=stdout, environ=environ)
            elif selected == 1:
                choice = _download(stdin=stdin, stdout=stdout)
            elif api_start <= selected < api_start + len(_api_providers()):
                choice = _api(
                    provider=_api_providers()[selected - api_start],
                    stdin=stdin,
                    stdout=stdout,
                    environ=environ,
                )
            elif selected == api_start + len(_api_providers()):
                choice = _other(detection=detection, stdin=stdin, stdout=stdout, environ=environ)
            elif selected == api_start + len(_api_providers()) + 1:
                choice = _sentence_transformer(detection=detection, stdin=stdin, stdout=stdout)
            else:
                choice = {"embed_provider": "none", "embed_model": ""}
        except prompts.Cancelled:
            if not prompts.interactive(stdin, stdout):
                raise ValueError("Setup ended before a complete embedding choice was made.")
            continue
        config = dict(settings.DEFAULTS)
        config.update({key: value for key, value in choice.items() if key in settings.DEFAULTS})
        code = settings.finish_setup(config, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
        if code == 0 or not prompts.interactive(stdin, stdout):
            return code
        stdout.write("Choose another source, or press Esc to return to the source list.\n")
