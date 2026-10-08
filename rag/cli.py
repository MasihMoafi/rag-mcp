from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

from rag import settings as user_settings
from rag.answer_contract import decode_public_answer
from rag.cli_agents import (
    AGENT_LABELS,
    available_agents as _available_agents,
    connect_agent as _connect_agent,
    connect_agy as _connect_agy,
    connect_claude as _connect_claude,
    connect_codex as _connect_codex,
    connect_copilot as _connect_copilot,
    connect_gemini as _connect_gemini,
    install_slash_commands as _install_slash_commands,
    offer_agent_connection as _offer_agent_connection,
    server_command as _server_command,
    show_generic_mcp_config as _show_generic_mcp_config,
)
from rag.cli_menu import MENU, menu as _menu


def _codex_status(environ: Mapping[str, str]) -> dict[str, object]:
    executable = shutil.which("codex", path=environ.get("PATH"))
    if executable is None:
        return {"available": False, "authenticated": False, "method": None}

    completed = subprocess.run(
        [executable, "login", "status"],
        capture_output=True,
        check=False,
        text=True,
        env=dict(environ),
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    authenticated = completed.returncode == 0 and "Logged in" in output
    method = "ChatGPT" if "ChatGPT" in output else ("API key" if authenticated else None)
    return {"available": True, "authenticated": authenticated, "method": method}


def _full_extra_installed() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _probe_embedding(config: Mapping[str, str]) -> int:
    from rag.embeddings import build_embedder, probe_dimensions

    return probe_dimensions(build_embedder(config))


def _should_offer_setup(environ: Mapping[str, str], stdin: TextIO) -> bool:
    """First interactive query with no saved choice: ask before searching."""
    if environ.get("RAG_MCP_EMBED_PROVIDER", "").strip():
        return False
    if user_settings.load_settings(environ):
        return False
    return bool(getattr(stdin, "isatty", lambda: False)())


_TOP_SHORTHAND = re.compile(r"^--?top-?(\d+)$")


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Accept `--top5` / `--top-5` as shorthand for `--top 5`."""
    normalized: list[str] = []
    for item in argv:
        match = _TOP_SHORTHAND.match(item)
        if match:
            normalized.extend(["--top", match.group(1)])
        else:
            normalized.append(item)
    return normalized




def _resolve_embedding_choice(model: str, provider: str | None, environ: Mapping[str, str], runtime: str | None = None) -> dict[str, str]:
    """Turn `rag config embedding <model>` into a full embedding config, guessing the provider from what runs here."""
    from rag import detect

    saved = user_settings.load_settings(environ)
    config = {**user_settings.DEFAULTS,
              **{key: value for key, value in saved.items()
                 if key in user_settings.DEFAULTS and isinstance(value, str)}}
    detection = detect.detect_all(environ)
    ollama_models = (detection.get("ollama") or {}).get("embedding_models") or []
    if runtime is not None:
        endpoint = dict(detect.LOCAL_OPENAI_RUNTIMES).get(runtime) if isinstance(runtime, str) else None
        if (provider is not None and provider != 'openai_compatible') or endpoint is None:
            raise ValueError('Choose a supported local OpenAI-compatible runtime.')
        if any(environ.get(name, '').strip() for name in
               ('RAG_MCP_EMBED_PROVIDER', 'RAG_MCP_EMBED_MODEL', 'RAG_MCP_EMBED_BASE_URL', 'RAG_MCP_EMBED_API_KEY')):
            raise ValueError('Embedding settings are controlled by the environment; change those overrides first.')
        available = any(isinstance(item, dict) and item.get('label') == runtime
                        and item.get('base_url') == endpoint
                        and model in (item.get('embedding_models') or [])
                        for item in (detection.get('local_runtimes') or []))
        if not available:
            raise ValueError('The model is not advertised by that local runtime. Check the runtime and retry.')
        config.update({'embed_provider': 'openai_compatible', 'embed_model': model,
                       'embed_base_url': endpoint, 'embed_api_key': ''})
        return config
    if provider is None:
        match = next((name for name in ollama_models if name == model or name.startswith(model + ":") or name.split(":")[0] == model), None)
        if match:
            provider, model = "ollama", match
        elif any(model.lower().startswith(prefix) for prefix in ("text-embedding-", "gemini-embedding", "qwen/", "voyage")):
            provider = "openai_compatible"
        elif "/" in model or model.lower() in {"all-minilm-l6-v2"}:
            provider = "fastembed"
        else:
            raise ValueError(
                f"Cannot tell where '{model}' runs. Add --provider ollama|fastembed|openai_compatible|sentencetransformer."
                + (f" Ollama has: {', '.join(ollama_models)}." if ollama_models else "")
            )
    config["embed_provider"] = provider
    config["embed_model"] = model
    if provider == "openai_compatible":
        family = "gemini" if "gemini" in model.lower() else "openrouter" if "/" in model else "openai"
        base_url, _ = detect.API_ENDPOINTS[family]
        config["embed_base_url"] = base_url
        if saved.get("embed_base_url") == base_url and saved.get("embed_api_key"):
            config["embed_api_key"] = str(saved["embed_api_key"])
        elif family not in detection.get("api_keys", []):
            raise ValueError(f"No API key for {family} in the environment. Run `rag setup` to enter one.")
        else:
            config["embed_api_key"] = ""
    return config


def _config_command(args: argparse.Namespace, *, stdout: TextIO, stderr: TextIO, environ: Mapping[str, str], verify) -> int:
    setting, value = args.setting, args.value.strip()
    runtime = getattr(args, 'runtime', None)
    if runtime is not None and setting != 'embedding':
        print('--runtime applies only to an embedding model.', file=stderr)
        return 2
    if setting == "show":
        for line in user_settings.describe_settings(environ):
            print(line, file=stdout)
        return 0
    if setting in user_settings.NUMBER_SETTINGS:
        key, low, high, _ = user_settings.NUMBER_SETTINGS[setting]
        if not value.isdigit() or not low <= int(value) <= high:
            print(f"Give a whole number from {low} to {high}, e.g. `rag config {setting} {low}`.", file=stderr)
            return 2
        user_settings.update_settings({key: value}, environ)
    elif setting == "answer":
        if not value or len(value) > 200 or any(ord(c) < 32 for c in value):
            print("Give an OpenRouter answer model ID.", file=stderr)
            return 2
        user_settings.update_settings({"answer_model": value}, environ)
    elif setting == "reranker":
        if not value:
            print("Give off, auto, fastembed, openrouter or a model name.", file=stderr)
            return 2
        if value in {"off", "auto"}:
            user_settings.update_settings({"reranker_type": value, "reranker_model": ""}, environ)
        elif value == "openrouter" or value.startswith(("qwen/qwen3-reranker-", "cohere/rerank-")):
            model = user_settings.RETRIEVAL_DEFAULTS["reranker_model"] if value == "openrouter" else value
            user_settings.update_settings({"reranker_type": "openrouter", "reranker_model": model}, environ)
        elif value == "fastembed" or value.startswith(("Xenova/", "jinaai/", "BAAI/bge-reranker")):
            model = user_settings.FASTEMBED_RERANKER if value == "fastembed" else value
            if verify is not None:
                from rag.embeddings import FastEmbedReranker

                print("Fetching the reranker model...", file=stdout)
                try:
                    scores = FastEmbedReranker(model).predict([("probe", "a passage")])
                except Exception as error:
                    print(f"Could not load the reranker: {error}\nNothing was saved.", file=stderr)
                    return 1
                print(f"Works: reranker answered {len(scores)} score(s).", file=stdout)
            user_settings.update_settings({"reranker_type": "fastembed", "reranker_model": model}, environ)
        else:
            user_settings.update_settings({"reranker_type": "cross-encoder", "reranker_model": value}, environ)
    elif setting == "contextual":
        mode = value.lower()
        if mode not in {"on", "off"}:
            print("Give on or off for contextual retrieval.", file=stderr)
            return 2
        user_settings.update_settings({"contextual_retrieval": mode}, environ)
    elif setting == "embedding":
        if not value:
            print("Give a model name, e.g. `rag config embedding qwen3-embedding:0.6b`.", file=stderr)
            return 2
        try:
            config = _resolve_embedding_choice(value, args.provider, environ, runtime)
        except ValueError as error:
            print(str(error), file=stderr)
            return 2
        code = user_settings.finish_setup(config, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
        if code != 0:
            return code
    for line in user_settings.describe_settings(environ)[:3]:
        print(line, file=stdout)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Answer questions from an explicit document scope using hybrid retrieval, reranking, and synthesis.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("setup", help="Choose where retrieval embeddings run, then choose a model")
    setup.add_argument("--provider", choices=user_settings.PROVIDERS, help="Skip the questions and set the provider directly")
    setup.add_argument("--model", default="", help="Embedding model name for the provider")
    setup.add_argument("--base-url", default="", dest="base_url", help="OpenAI-compatible endpoint URL")
    setup.add_argument("--api-key", default="", dest="api_key", help="API key for openai_compatible")
    setup.add_argument("--ollama-url", default="", dest="ollama_url", help="Ollama URL (default http://localhost:11434)")
    setup.add_argument("--no-verify", action="store_true", dest="no_verify", help="Save without embedding a test sentence")
    setup.add_argument("--auto", action="store_true", help="Pick the best option this machine already has, no questions")
    commands.add_parser("menu", help="open the interactive human terminal UI")

    config = commands.add_parser("config", help="Show or change settings: top N, reranker, contextual retrieval, embedding model")
    config.add_argument("setting", nargs="?", choices=["show", *user_settings.NUMBER_SETTINGS, "answer", "reranker", "contextual", "embedding"], default="show")
    config.add_argument("value", nargs="?", default="")
    config.add_argument("--provider", choices=user_settings.PROVIDERS, help="Where the embedding model runs (guessed when omitted)")
    config.add_argument("--runtime", help="Detected local OpenAI-compatible runtime (LM Studio or vLLM) for an embedding model")
    config.add_argument("--no-verify", action="store_true", dest="no_verify")

    doctor = commands.add_parser("doctor", help="Check local RAG, embedding and MCP-app setup")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    model_capabilities = commands.add_parser("capabilities", help="Show configured engine stages and model readiness")
    model_capabilities.add_argument("--json", action="store_true", dest="as_json")
    local_models = commands.add_parser("models", help="Discover configured and locally installed models on demand")
    local_models.add_argument("--json", action="store_true", dest="as_json")

    query = commands.add_parser("query", help="Answer a question from a supported document or folder")
    query.add_argument("question")
    query_scope = query.add_mutually_exclusive_group(required=True)
    query_scope.add_argument("--path")
    query_scope.add_argument("--workspace", help="query a workspace shared with Studio and MCP")
    query.add_argument("--conversation", metavar="ID", help="continue and save to a shared workspace conversation")
    query.add_argument(
        "--type",
        dest="extensions",
        action="append",
        help="limit a directory to this file type; repeat for more than one (for example --type md --type pdf)",
    )
    query.add_argument("--top", type=int, default=None, metavar="N", help="How many reranked passages synthesis may use; default 5")
    query.add_argument(
        "--format",
        dest="output_format",
        choices=("auto", "text", "json"),
        default="auto",
        help="output format (default: answer text; terminals also show progress)",
    )
    search = commands.add_parser("search", help="Retrieve passages from a shared workspace without generating an answer")
    search.add_argument("question")
    search.add_argument("--workspace", required=True, help="Workspace shared with Studio and MCP")
    search.add_argument("--source", action="append", dest="source_ids", help="Restrict to a source ID; repeat for several")
    search.add_argument("--revision", help="Reject the search if the workspace changed")
    search.add_argument("--json", action="store_true", dest="as_json")
    search.add_argument("--no-content", action="store_true", help="Omit private passage text and scores")
    connect = commands.add_parser("connect", help="Register the MCP server with an installed MCP app")
    connect.add_argument("agent", choices=["codex", "claude", "copilot", "gemini", "agy"])
    connect.add_argument("--scope", choices=["user", "project"], default="user", help="Configuration scope where the app supports it")
    connect.add_argument(
        "--install-prompt",
        action="store_true",
        help="also install the optional slash prompt for supported agents",
    )
    connect_codex = commands.add_parser("connect-codex", help="Same as `rag connect codex`")
    connect_codex.add_argument("--install-prompt", action="store_true")
    serve = commands.add_parser("serve", help="Run the local Studio companion")
    serve.add_argument("--port", type=int, default=47831)
    serve.add_argument("--studio-url", default="https://rag.masihmoafi.com", dest="studio_url")
    studio = commands.add_parser("studio", help="Open the bundled local Studio and companion")
    studio.add_argument("--port", type=int, default=47831)
    studio.add_argument("--web-port", type=int, default=3000, help="Loopback port for the bundled Studio (default 3000)")
    studio.add_argument("--no-open", action="store_true")
    studio.add_argument("--studio-url", default=None, dest="studio_url", help="Pair a separately running Studio instead of launching the bundle")
    commands.add_parser("mcp", help="Run the stdio MCP server")

    runs = commands.add_parser("runs", help="Inspect private local CLI/MCP run traces")
    runs.add_argument("trace_id", nargs="?", help="Run ID to inspect")
    runs.add_argument("--limit", type=int, default=20, help="Recent runs to list (1–100)")
    runs.add_argument("--offset", type=int, default=0, help="Skip this many recent records (0–10000)")
    runs.add_argument("--json", action="store_true", dest="as_json")
    runs.add_argument("--result", action="store_true", help="Show the saved answer and source evidence for one run")
    workspaces = commands.add_parser("workspaces", help="List document workspaces shared with Studio and MCP")
    workspaces.add_argument("workspace_id", nargs="?", help="Show source IDs and revision for one workspace")
    workspaces.add_argument("--json", action="store_true", dest="as_json")
    sources = commands.add_parser("sources", help="Import local files into a shared workspace without indexing")
    sources.add_argument("action", choices=("add",))
    sources.add_argument("workspace_id", help="Shared workspace ID")
    sources.add_argument("--path", required=True, help="Supported file or deliberately scoped folder")
    sources.add_argument("--type", dest="extensions", action="append", help="Limit a folder to this file type; repeat as needed")
    sources.add_argument("--json", action="store_true", dest="as_json")
    chunks = commands.add_parser("chunks", help="Preview or inspect published chunks for one saved workspace source")
    chunks.add_argument("workspace_id", help="Shared workspace ID")
    chunks.add_argument("source_id", help="Source ID in that workspace")
    chunks.add_argument("--revision", help="Expected workspace revision; reject a stale source")
    chunks.add_argument("--limit", type=int, default=50, help="Chunks per page (1–100)")
    chunks.add_argument("--offset", type=int, default=0, help="First chunk ordinal in this page")
    chunks.add_argument("--content", action="store_true", help="Explicitly show private text for this page's chunks")
    chunks.add_argument("--view", choices=["preview", "indexed"], default="preview", help="Next-build preview or published index metadata")
    chunks.add_argument("--scope-source", action="append", dest="scope_source_ids", help="Source ID in the published query scope; repeat to inspect a selected subset")
    chunks.add_argument("--json", action="store_true", dest="as_json")
    evaluations = commands.add_parser("evaluations", help="Read or edit a workspace's shared evaluation questions")
    evaluations.add_argument("workspace_id", help="Shared workspace ID")
    evaluations.add_argument("--json", action="store_true", dest="as_json")
    question_edit = evaluations.add_mutually_exclusive_group()
    question_edit.add_argument("--add", metavar="QUESTION", help="Add a question")
    question_edit.add_argument("--remove", metavar="ID", help="Remove a question by ID")
    question_edit.add_argument("--run", action="store_true", help="Run the question set through shared engine retrieval, without answer synthesis")
    question_edit.add_argument("--runs", action="store_true", help="List saved evaluation batches")
    question_edit.add_argument("--batch", metavar="ID", help="Read one saved evaluation batch")
    evaluations.add_argument("--source", action="append", dest="source_ids", help="Source ID to include with --run; repeat for multiple sources")
    evaluations.add_argument("--limit", type=int, default=20, help="Saved batches per page (1–100)")
    evaluations.add_argument("--offset", type=int, default=0, help="Saved batches to skip (0–10000)")
    labels = commands.add_parser("labels", help="Read or mark human relevance for a saved evaluation batch")
    labels.add_argument("workspace_id", help="Shared workspace ID")
    labels.add_argument("batch_id", help="Saved evaluation batch ID")
    labels.add_argument("--set", metavar="RUN_ID:RANK:LABEL", help="Mark a passage relevant, irrelevant, or clear its label")
    labels.add_argument("--scores", action="store_true", help="Score the batch against passages marked relevant")
    labels.add_argument("--json", action="store_true", dest="as_json")
    conversations = commands.add_parser("conversations", help="List, read, or append to shared workspace conversations")
    conversations.add_argument("workspace_id", help="Shared workspace ID")
    conversations.add_argument("conversation_id", nargs="?", help="Conversation ID to read or append to")
    conversations.add_argument("--append-run", metavar="RUN_ID", help="Append a saved successful answer run")
    conversations.add_argument("--question", help="The question asked in the saved run")
    conversations.add_argument("--limit", type=int, default=20, help="Conversations per page (1–100)")
    conversations.add_argument("--offset", type=int, default=0, help="Conversations to skip (0–10000)")
    conversations.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
    query_fn: Callable[[str, str], str] | None = None,
    embedding_probe: Callable[[Mapping[str, str]], int] | None = None,
    token_factory: Callable[[], str] = lambda: secrets.token_urlsafe(32),
    server_factory: Callable[..., object] | None = None,
    open_url: Callable[[str], object] = webbrowser.open,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    environ = environ or os.environ
    raw = list(argv) if argv is not None else sys.argv[1:]
    if not raw:
        _parser().print_help(stdout)
        return 0
    args = _parser().parse_args(_normalize_argv(raw))

    if args.command == "capabilities":
        from rag.control import capabilities
        try:
            value = capabilities()
        except (ValueError, OSError) as error:
            print(f"Cannot read capabilities: {error}", file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        else:
            for stage in value['stages']:
                print(f"{stage['stage']:<12} {stage['availability']:<11} {stage['provider']} · {stage['model'] or '—'}", file=stdout)
        return 0

    if args.command == "models":
        from rag.control import model_inventory
        try:
            value = model_inventory()
        except (ValueError, OSError) as error:
            print(f"Cannot discover models: {error}", file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        else:
            for item in value['models']:
                marker = ' [configured]' if item['configured'] else ''
                print(f"{item['stage']:<12} {item['provider']:<20} {item['model']}{marker}", file=stdout)
            for source in value['sources']:
                if source['status'] != 'complete':
                    print(f"{source['source']}: {source['status']}", file=stderr)
        return 0

    if args.command == "runs":
        from rag.trace_reader import list_traces, read_trace

        if args.result and not args.trace_id:
            print("--result requires a run ID.", file=stderr)
            return 2

        try:
            if args.result:
                from rag.run_results import RunResultStore
                value = RunResultStore().read(args.trace_id)
            else:
                value = read_trace(args.trace_id) if args.trace_id else list_traces(limit=args.limit, offset=args.offset)
        except (ValueError, OSError) as error:
            print(f"Cannot read runs: {error}", file=stderr)
            return 2
        if value is None:
            print("Run not found or unavailable. Saved answers expire after seven days; use rag runs to list metadata traces.", file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        elif args.result:
            print(f"Run {args.trace_id}: {value['status']}", file=stdout)
            print(value.get('answer') or value.get('error') or 'No answer saved.', file=stdout)
            for passage in value.get('evidence', []):
                print(f"\n[{passage['rank']}] {passage['name']}\n{passage['content']}", file=stdout)
        elif args.trace_id:
            print(f"Run {args.trace_id}", file=stdout)
            for span in value["spans"]:
                label = span.get("error_class") or span["attributes"].get("model") or ""
                print(f"{span['operation']:<28} {span['status']:<10} {span['duration_ms']:>8.1f} ms  {label}", file=stdout)
            cost = value["accounting"]["reported_cost_subtotal_usd"]
            label = f"${cost:.6f}" if cost is not None else "unknown"
            print(f"Reported cost subtotal: {label}. Use --json for recorded attributes and passage ranks.", file=stdout)
        elif not value["traces"]:
            print("No completed metadata runs." if value["enabled"] else "Tracing is disabled.", file=stdout)
        else:
            print("RUN ID                            STATUS       DURATION   REPORTED COST SUBTOTAL", file=stdout)
            for run in value["traces"]:
                cost = run["reported_cost_subtotal_usd"]
                label = f"${cost:.6f}" if cost is not None else "unknown"
                print(f"{run['trace_id']}  {run['status']:<10} {run['duration_ms']:>8.1f} ms  {label}", file=stdout)
        if not args.as_json and value.get("skipped"):
            print(f"Skipped {value['skipped']} unavailable records.", file=stderr)
        return 0

    if args.command == "conversations":
        from filelock import Timeout
        from rag.conversations import ConversationStore
        from rag.workspaces import WorkspaceConflict, WorkspaceStore

        store = ConversationStore(WorkspaceStore.from_environ(environ))
        try:
            if bool(args.append_run) != bool(args.question) or args.append_run and not args.conversation_id:
                raise ValueError('--append-run requires a conversation ID and --question.')
            if args.append_run:
                try:
                    previous = store.get(args.workspace_id, args.conversation_id)
                except FileNotFoundError:
                    previous = None
                value = store.append(args.workspace_id, args.conversation_id, args.append_run, args.question,
                                     expected_revision=previous['revision'] if previous else None)
            elif args.conversation_id:
                value = store.get(args.workspace_id, args.conversation_id)
            else:
                value = store.list(args.workspace_id, limit=args.limit, offset=args.offset)
        except (ValueError, OSError, WorkspaceConflict, Timeout) as error:
            print(f'Cannot access conversations: {error}', file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        elif args.conversation_id:
            print(f"Conversation {value['conversation_id']} · {len(value['turns'])} turns", file=stdout)
            for turn in value['turns']:
                print(f"\n{turn['run_id']}  {turn['question']}\n{turn['answer']}", file=stdout)
        else:
            for item in value['conversations']:
                print(f"{item['conversation_id']}  {item['turn_count']} turns  {item['last_question']}", file=stdout)
        return 0

    if args.command == "labels":
        from filelock import Timeout
        from rag.evaluation_labels import EvaluationLabelStore
        from rag.workspaces import WorkspaceConflict, WorkspaceStore

        store = EvaluationLabelStore(WorkspaceStore.from_environ(environ))
        try:
            value = store.get(args.workspace_id, args.batch_id)
            if args.set:
                parts = args.set.rsplit(':', 2)
                if len(parts) != 3 or not parts[1].isdigit() or parts[2] not in {'relevant', 'irrelevant', 'clear'}:
                    raise ValueError('Use RUN_ID:RANK:relevant, irrelevant, or clear.')
                value = store.set(args.workspace_id, args.batch_id, parts[0], int(parts[1]),
                                  None if parts[2] == 'clear' else parts[2], expected_revision=value['revision'])
            if args.scores:
                from rag.evaluation_scores import score_batch
                value = score_batch(args.workspace_id, args.batch_id, store)
        except (ValueError, OSError, WorkspaceConflict, Timeout) as error:
            print(f'Cannot access evaluation labels: {error}', file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        elif args.scores:
            summary = value['summary']
            if not summary['judged_questions']:
                print('No question has a passage marked relevant yet.', file=stdout)
            else:
                print('  '.join(f"{name} {summary[key]:.2f}" for name, key in (
                    ('hit rate', 'hit_rate'), ('recall', 'recall'), ('MRR', 'mrr'), ('nDCG', 'ndcg'), ('precision', 'precision'))), file=stdout)
            print(f"{summary['judged_questions']} judged questions, {summary['unjudged_questions']} without right answers, "
                  f"{summary['unjudged_passages']} unlabelled passages, {summary['missing_evidence']} with expired evidence", file=stdout)
        elif not value['labels']:
            print('No human relevance labels in this batch.', file=stdout)
        else:
            for item in value['labels']:
                print(f"{item['run_id']}:{item['rank']}  {item['label']}  {item['source_id']}", file=stdout)
        return 0

    if args.command == "evaluations":
        from rag.evaluation_sets import EvaluationSetStore
        from rag.evaluation_runs import EvaluationRunStore, execute_evaluation
        from rag.workspaces import WorkspaceConflict, WorkspaceStore
        from filelock import Timeout

        store = EvaluationSetStore(WorkspaceStore.from_environ(environ))
        try:
            if args.source_ids is not None and not args.run:
                raise ValueError('--source requires --run.')
            if args.run or args.runs or args.batch:
                batches = EvaluationRunStore(store.workspaces)
                if args.run:
                    with contextlib.redirect_stdout(stderr):
                        value = execute_evaluation(args.workspace_id, source_ids=args.source_ids, store=batches, client='cli')
                else:
                    value = batches.get(args.workspace_id, args.batch) if args.batch else batches.list(
                        args.workspace_id, limit=args.limit, offset=args.offset)
                if args.as_json:
                    print(json.dumps(value, ensure_ascii=False), file=stdout)
                elif args.runs:
                    for batch in value['batches']:
                        print(f"{batch['batch_id']}  {batch['status']}  {batch['completed_count']}/{batch['question_count']} checked  {batch['passage_count']} recorded passages", file=stdout)
                    if not value['batches']:
                        print('No saved evaluation batches.', file=stdout)
                    if value['next_offset'] is not None:
                        print(f"More batches: --offset {value['next_offset']}", file=stdout)
                    if value['skipped']:
                        print(f"Skipped {value['skipped']} unavailable batches.", file=stderr)
                else:
                    print(f"{value['batch_id']}  {value['status']}  {len(value['rows'])}/{len(value['questions'])} checked", file=stdout)
                    for question, row in zip(value['questions'], value['rows']):
                        print(f"{row['run_id']}  {row['status']}  {row['passage_count']} passages  {question['question']}", file=stdout)
                    if value['error']:
                        print(value['error'], file=stderr)
                return (0 if value['status'] == 'success' else 130 if value['status'] == 'cancelled' else 1) if args.run else 0
            value = store.get(args.workspace_id)
            if args.add is not None:
                question = " ".join(args.add.split())
                if any(item['question'].casefold() == question.casefold() for item in value['questions']):
                    raise ValueError("That question is already in this set.")
                value = store.replace(args.workspace_id, [*value['questions'], {'id': secrets.token_hex(16), 'question': question}],
                                      expected_revision=value['revision'])
            elif args.remove is not None:
                if not any(item['id'] == args.remove for item in value['questions']):
                    raise ValueError("Question ID not found.")
                value = store.replace(args.workspace_id, [item for item in value['questions'] if item['id'] != args.remove],
                                      expected_revision=value['revision'])
        except (ValueError, OSError, WorkspaceConflict, Timeout) as error:
            print(f"Cannot access evaluations: {error}", file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        elif not value['questions']:
            print("No evaluation questions in this workspace.", file=stdout)
        else:
            for item in value['questions']:
                print(f"{item['id']}  {item['question']}", file=stdout)
        return 0

    if args.command == "sources":
        from filelock import Timeout

        try:
            with contextlib.redirect_stdout(stderr):
                from rag.workspace_import import import_workspace_sources
                from rag.workspaces import WorkspaceStore
                value = import_workspace_sources(WorkspaceStore.from_environ(environ), args.workspace_id,
                                                 args.path, args.extensions)
        except (ValueError, OSError, UnicodeError, ImportError, Timeout) as error:
            print(f"Cannot import sources: {error}", file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        else:
            print(f"Imported {value['imported_count']} sources into {value['workspace_id']} · "
                  f"{value['document_count']} total · revision {value['revision']}", file=stdout)
        return 0

    if args.command == "workspaces":
        from rag.workspaces import WorkspaceStore

        try:
            store = WorkspaceStore.from_environ(environ)
            value = store.describe(args.workspace_id) if args.workspace_id else store.list_workspaces()
        except (ValueError, OSError) as error:
            print(f"Cannot read workspaces: {error}", file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value if args.workspace_id else {"schema_version": 1, "workspaces": value}, ensure_ascii=False), file=stdout)
        elif args.workspace_id:
            print(f"{value['name']} · {value['id']} · revision {value['revision']}", file=stdout)
            for document in value['documents']:
                print(f"{document['id']}  {document['name']}  {document['size_bytes']} B", file=stdout)
        elif not value:
            print("No shared workspaces. Pair Studio with `rag studio` and add a source.", file=stdout)
        else:
            print("WORKSPACE                 DOCUMENTS   SIZE       NAME", file=stdout)
            for workspace in value:
                print(
                    f"{workspace['id']:<25} {workspace['document_count']:>9}   "
                    f"{workspace['total_bytes']:>8} B  {workspace['name']}",
                    file=stdout,
                )
        return 0

    if args.command == "chunks":
        from rag.workspaces import WorkspaceStore

        try:
            if args.scope_source_ids is not None and args.view != "indexed":
                raise ValueError("--scope-source requires --view indexed.")
            store = WorkspaceStore.from_environ(environ)
            if args.view == "indexed":
                from rag.index_inspection import inspect_published_chunks
                revision = args.revision or store.get(args.workspace_id)["revision"]
                value = inspect_published_chunks(store, args.workspace_id, args.source_id, revision,
                                                 limit=args.limit, offset=args.offset, include_content=args.content,
                                                 scope_source_ids=args.scope_source_ids)
            else:
                from rag.chunk_preview import inspect_workspace_chunks
                value = inspect_workspace_chunks(store, args.workspace_id, args.source_id, args.revision,
                                                 limit=args.limit, offset=args.offset, include_content=args.content)
        except (ValueError, OSError) as error:
            print(f"Cannot inspect chunks: {error}", file=stderr)
            return 2
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        elif args.view == "indexed":
            print(f"{value['name']} · published index: {value['status']} · workspace revision {value['revision']}", file=stdout)
            print("Published document metadata; vector storage is not verified by this read.", file=stdout)
            for item in value['chunks']:
                location = f"{item['start_offset']}–{item['end_offset']}" if item['start_offset'] is not None else "unlocated"
                print(f"[{item['ordinal'] + 1}] {location}" + (f"  {item['heading']}" if args.content and item['heading'] else ""), file=stdout)
                if args.content:
                    print(item['indexed_content'], file=stdout)
            if value['next_offset'] is not None:
                print(f"More chunks: --offset {value['next_offset']}", file=stdout)
        else:
            print(f"{value['name']} · {value['total_chunks']} engine preview chunks · workspace revision {value['revision']}", file=stdout)
            print(f"Size {value['chunk_size']} · overlap {value['chunk_overlap']} · existing index may differ", file=stdout)
            for item in value['chunks']:
                location = f"{item['start_offset']}–{item['end_offset']}" if item['start_offset'] is not None else "unlocated"
                print(f"[{item['ordinal'] + 1}] {location}" + (f"  {item['heading']}" if args.content and item['heading'] else ""), file=stdout)
                if args.content:
                    print(item['content'], file=stdout)
            if value['next_offset'] is not None:
                print(f"More chunks: --offset {value['next_offset']}", file=stdout)
        return 0

    if args.command == "menu":
        return _menu(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, embedding_probe=embedding_probe,
                     token_factory=token_factory, server_factory=server_factory, open_url=open_url)

    if args.command == "setup":
        verify = None if args.no_verify else (embedding_probe or _probe_embedding)
        if args.auto:
            return user_settings.auto_setup(stdout=stdout, stderr=stderr, environ=environ, verify=verify)
        if args.provider is None:
            code = user_settings.run_setup(stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, verify=verify)
            return code
        if args.provider == "openai_compatible" and not args.api_key:
            print("--api-key is required for openai_compatible.", file=stderr)
            return 2
        config = dict(user_settings.DEFAULTS)
        config.update(
            {
                "embed_provider": args.provider,
                "embed_model": args.model or user_settings.default_model(args.provider),
                "embed_base_url": args.base_url,
                "embed_api_key": args.api_key,
            }
        )
        if args.ollama_url:
            config["ollama_url"] = args.ollama_url
        return user_settings.finish_setup(config, stdout=stdout, stderr=stderr, environ=environ, verify=verify)

    if args.command == "config":
        verify = None if args.no_verify else (embedding_probe or _probe_embedding)
        return _config_command(args, stdout=stdout, stderr=stderr, environ=environ, verify=verify)

    if args.command == "doctor":
        from rag import detect

        embedding = user_settings.embedding_config(environ)
        detection = detect.detect_all(environ, embedding["ollama_url"])
        report = {
            "detected": detection,
            "python": {"available": True, "version": sys.version.split()[0]},
            "codex": _codex_status(environ),
            "mcp_apps": {key: bool(shutil.which(executable, path=environ.get("PATH"))) for key, _, executable in AGENT_LABELS},
            "embedding": {
                "provider": embedding["embed_provider"],
                "model": embedding["embed_model"],
                "configured": user_settings.has_embedding(embedding),
                "config_path": str(user_settings.config_path(environ)),
                "full_extra": _full_extra_installed(),
            },
        }
        if args.as_json:
            print(json.dumps(report), file=stdout)
        else:
            codex = report["codex"]
            status = "signed in" if codex["authenticated"] else "not signed in"
            print(
                f"Python {report['python']['version']} · Codex {status} · "
                f"Embedding: {user_settings.describe_embedding(embedding)}",
                file=stdout,
            )
            for line in detect.describe(detection):
                print(f"Detected: {line}", file=stdout)
            apps = [label for key, label, _ in AGENT_LABELS if report["mcp_apps"][key]]
            if apps:
                print("MCP apps available: " + ", ".join(apps), file=stdout)
            if not report["embedding"]["configured"]:
                print("Run `rag setup` (or `rag setup --auto`) to add meaning-based search.", file=stdout)
        # MCP hosts are optional integrations, not prerequisites for local RAG.
        return 0

    if args.command == "connect-codex":
        return _connect_codex(environ, stdout, stderr, install_prompt=args.install_prompt)
    if args.command == "connect":
        return _connect_agent(
            args.agent,
            environ,
            stdout,
            stderr,
            scope=args.scope,
            install_prompt=args.install_prompt,
        )

    if args.command in {"serve", "studio"}:
        if server_factory is None:
            from rag.companion import create_companion_server

            server_factory = create_companion_server
        with contextlib.ExitStack() as stack:
            studio_url = args.studio_url
            if args.command == 'studio' and not studio_url:
                from rag.studio_bundle import StudioBundleError, serve_studio
                try:
                    studio_url = stack.enter_context(serve_studio(args.web_port, environ=environ))
                except (StudioBundleError, OSError) as error:
                    print(f"Could not start the bundled Studio: {error}", file=stderr)
                    return 1
            token = token_factory()
            try:
                parts = urlsplit(studio_url)
                if parts.scheme not in {'http', 'https'} or not parts.netloc or parts.username or parts.password:
                    print('Studio URL must be an HTTP or HTTPS origin.', file=stderr)
                    return 2
                origins = {'allowed_origins': {f'{parts.scheme}://{parts.netloc}'}} if args.command == 'studio' else {}
                server = server_factory(token=token, port=args.port, **origins)
            except OSError as error:
                print(f"Could not start the local companion: {error}", file=stderr)
                return 1
            url = f"{studio_url.rstrip('/')}/studio#local={token}"
            if server.server_port != 47831:
                url += f"&port={server.server_port}"
            print("Starting the local companion for Studio.", file=stdout)
            print("Keep this terminal open while Studio uses your local documents and models.", file=stdout)
            print(f"Local companion: http://127.0.0.1:{server.server_port}", file=stdout)
            print(f"Embedding: {user_settings.describe_embedding(user_settings.embedding_config(environ))}", file=stdout)
            if args.command == "studio":
                print(f"Opening the local Studio in your browser: {url}", file=stdout)
                print("Press Ctrl-C in this terminal to stop Studio and its companion.", file=stdout)
            else:
                print(f"Studio pairing URL: {url}", file=stdout)
                print("Open that URL in a browser. Press Ctrl-C here to stop the companion.", file=stdout)
            try:
                if args.command == "studio" and not args.no_open:
                    open_url(url)
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        return 0

    if args.command == "mcp":
        from rag.server import main as mcp_main

        mcp_main()
        return 0

    if args.command == "search":
        from rag.execution import execute_run, retrieval_response
        from rag.workspaces import WorkspaceStore

        with contextlib.redirect_stdout(stderr):
            result = execute_run(args.question, workspace_id=args.workspace, client='cli', mode='retrieve',
                                 workspace_store=WorkspaceStore.from_environ(environ),
                                 source_ids=args.source_ids, workspace_revision=args.revision)
        value = retrieval_response(result, include_content=not args.no_content)
        if args.as_json:
            print(json.dumps(value, ensure_ascii=False), file=stdout)
        elif value.get('error'):
            print(f"Search failed ({value['run_id']}): {value['error']}", file=stderr)
        else:
            print(f"{value['passage_count']} passages · run {value['run_id']}", file=stdout)
            for passage in value['passages']:
                print(f"\n{passage['rank']}. {passage['name']} [{passage['source_id']}]", file=stdout)
                if not passage['location_verified']:
                    print('Source offset unavailable; this passage cannot be opened at an exact location.', file=stdout)
                if not args.no_content:
                    print(passage['content'], file=stdout)
        return 1 if value.get('error') else 0

    if args.conversation and (not args.workspace or query_fn is not None or args.extensions):
        print('--conversation requires --workspace and the local engine query path, without --type.', file=stderr)
        return 2
    if args.workspace:
        from rag.workspaces import WorkspaceStore

        try:
            path = WorkspaceStore.from_environ(environ).resolve(args.workspace)
        except (ValueError, OSError) as error:
            print(f"Cannot open workspace: {error}", file=stderr)
            return 2
    else:
        path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"Path does not exist: {path}", file=stderr)
        return 2
    conversation_output = None
    if query_fn is None:
        if not args.workspace:
            # Refuse an unsafe folder before setup can probe or download a model.
            from rag.server import _validate_rag_scope
            try:
                scope = _validate_rag_scope(path, args.extensions)
                if scope is not None and not scope.files:
                    raise ValueError("No readable documents found in the selected scope.")
            except (ValueError, OSError) as error:
                message = str(error)
                if path.is_dir() and "token limit" in message:
                    message += (
                        " For mixed-format folders, filter a deliberate source type"
                        " with --type md (or another matching extension)."
                    )
                print(message, file=stderr)
                return 2
        if _should_offer_setup(environ, stdin):
            code = user_settings.run_setup(
                stdin=stdin, stdout=stdout, stderr=stderr, environ=environ, verify=embedding_probe or _probe_embedding
            )
            if code != 0:
                return code
        from rag.execution import execute_run, public_response

        def query_fn(question, document_path, extensions=None):
            nonlocal conversation_output
            if args.conversation:
                from rag.conversations import ConversationStore, run_conversation_query
                from rag.workspaces import WorkspaceStore
                try:
                    conversation_output = run_conversation_query(question, args.workspace, args.conversation,
                        client='cli', store=ConversationStore(WorkspaceStore.from_environ(environ)))
                except ValueError as error:
                    conversation_output = {'error': str(error)}
                except OSError:
                    conversation_output = {'error': 'Conversation storage is unavailable.'}
                return json.dumps(conversation_output, ensure_ascii=False)
            return public_response(execute_run(
                question, "" if args.workspace else document_path, extensions,
                args.workspace or "", client="cli",
            ))
    from rag.prompts import interactive

    human_output = args.output_format == "text" or (
        args.output_format == "auto" and interactive(stdin, stdout)
    )
    previous = {name: os.environ.get(name) for name in ("RAG_MCP_RERANK_TOP_K", "RAG_MCP_TOP_K")}
    try:
        if args.top is not None:
            os.environ["RAG_MCP_RERANK_TOP_K"] = str(max(1, args.top))
            os.environ["RAG_MCP_TOP_K"] = str(max(args.top, int(os.environ.get("RAG_MCP_TOP_K") or 5)))
        if human_output:
            print(f"Searching {path.name}…", file=stderr, flush=True)
            # Native database diagnostics also write directly to fd 2.
            with tempfile.TemporaryFile(mode="w+") as diagnostics:
                original_stderr = os.dup(2)
                try:
                    os.dup2(diagnostics.fileno(), 2)
                    with contextlib.redirect_stderr(diagnostics):
                        result = (
                            query_fn(args.question, str(path), args.extensions)
                            if args.extensions else query_fn(args.question, str(path))
                        )
                finally:
                    os.dup2(original_stderr, 2)
                    os.close(original_stderr)
        else:
            # Machine-readable stdout must contain one JSON value. Retrieval
            # progress remains available on stderr for operators and logs.
            with contextlib.redirect_stdout(stderr):
                result = (
                    query_fn(args.question, str(path), args.extensions)
                    if args.extensions else query_fn(args.question, str(path))
                )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if conversation_output is not None:
        is_error = bool(conversation_output.get('error'))
        public_text = conversation_output.get('error') if is_error else conversation_output.get('result')
        if not isinstance(public_text, str):
            public_text, is_error = 'Conversation query returned an invalid response.', True
    else:
        public_text, is_error = decode_public_answer(result)
    display = (
        json.dumps(conversation_output if conversation_output is not None else {"error" if is_error else "result": public_text}, ensure_ascii=False)
        if args.output_format == "json"
        else public_text
    )
    print(display, file=stdout)
    if is_error:
        return 1
    if conversation_output and conversation_output.get('conversation_error'):
        print(conversation_output['conversation_error'], file=stderr)
        return 3
    if not user_settings.has_embedding(user_settings.embedding_config(environ)):
        print(user_settings.BM25_ONLY_NOTICE, file=stderr)
    return 0


def main(argv: Sequence[str] | None = None, **kwargs: Any) -> int:
    from rag.prompts import Cancelled

    try:
        return _main(argv, **kwargs)
    except Cancelled:
        return 130
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=kwargs.get("stdout") or sys.stdout)
        return 130
    except ValueError as error:
        print(str(error), file=kwargs.get("stderr") or sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
