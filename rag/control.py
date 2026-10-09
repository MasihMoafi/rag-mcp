"""Credential-free control-plane views over the CLI/MCP settings store."""
import os
from urllib.parse import urlsplit

from rag import settings


def read_config():
    with settings.locked_settings(create=False), settings.frozen_settings():
        embedding = settings.embedding_config()
        retrieval = settings.retrieval_config()
        revision = settings.config_revision()
    return {'schema_version': 1,
            'revision': revision,
            'embedding': {'provider': embedding['embed_provider'], 'model': embedding['embed_model']},
            'retrieval': {key: retrieval[key] for key in settings.RETRIEVAL_DEFAULTS},
            'overrides': [key for key, name in {**settings.ENV_KEYS, **settings.RETRIEVAL_ENV}.items()
                          if os.environ.get(name, '').strip() and key != 'embed_api_key']}


def patch_config(changes):
    fields = {'context_radius': ['context_radius'], 'top': ['top_k'], 'candidates': ['candidate_k'], 'answer_model': ['answer_model'],
              'reranker': ['reranker_type', 'reranker_model'], 'contextual_retrieval': ['contextual_retrieval'],
              'embedding_provider': ['embed_provider'], 'embedding_model': ['embed_model'],
              'embedding_source': [], 'embedding_runtime': ['embed_base_url', 'embed_api_key']}
    if not isinstance(changes, dict) or not (set(changes) - {'revision'}) or set(changes) - (fields.keys() | {'revision'}):
        raise ValueError('Unsupported engine setting. Credentials are configured with rag setup in a terminal.')
    expected_revision = changes.get('revision', settings.REVISION_UNSET)
    if expected_revision is not settings.REVISION_UNSET:
        if expected_revision is not None and (not isinstance(expected_revision, str)
                or len(expected_revision) != 64 or any(char not in '0123456789abcdef' for char in expected_revision)):
            raise ValueError('Invalid engine configuration revision.')
        if expected_revision != settings.config_revision():
            raise settings.SettingsConflict('Engine settings changed. Reload before saving.')
    overridden = read_config()['overrides']
    embedding_choice = 'embedding_provider' in changes or 'embedding_model' in changes
    if embedding_choice:
        provider = changes.get('embedding_provider')
        model = changes.get('embedding_model')
        source = changes.get('embedding_source')
        runtime = changes.get('embedding_runtime')
        if expected_revision is settings.REVISION_UNSET:
            raise ValueError('Reload engine settings before changing the embedding model.')
        if provider == 'none':
            if model is not None or source is not None or runtime is not None:
                raise ValueError('Disable embeddings without a model choice.')
        elif (not isinstance(provider, str)
              or provider not in {'ollama', 'fastembed', 'sentencetransformer', 'openai_compatible'}
              or not isinstance(model, str) or not model.strip() or len(model) > 200
              or any(ord(char) < 32 for char in model)
              or not isinstance(source, str)
              or source not in {'ollama', 'fastembed_cache', 'sentence_transformers_cache', 'local_runtime'}
              or not isinstance(runtime, str)
              or (source == 'local_runtime') != (provider == 'openai_compatible')):
            raise ValueError('Choose a discovered local embedding model.')
        if provider != 'none':
            found = next((item for item in model_inventory()['models']
                          if item['stage'] == 'embedding' and item['provider'] == provider
                          and item['model'] == model and item['location'] == 'local'
                          and item['discovery_source'] == source and item['runtime'] == runtime), None)
            if found is None:
                raise ValueError('That local model is no longer available. Find local models again.')
    for key, value in changes.items():
        if key == 'revision':
            continue
        if any(field in overridden for field in fields[key]):
            raise ValueError('This setting is controlled by the engine environment. Change it there and restart the engine.')
        if key in settings.NUMBER_SETTINGS:
            _, low, high, _ = settings.NUMBER_SETTINGS[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be a whole number from {low} to {high}.')
        elif key in {'embedding_source', 'embedding_runtime', 'embedding_provider', 'embedding_model'}:
            continue
        elif not isinstance(value, str) or not value.strip() or len(value) > 200 or any(ord(c) < 32 for c in value):
            raise ValueError('Setting must be a non-empty model name or supported option.')
        elif key == 'contextual_retrieval' and value not in {'on', 'off'}:
            raise ValueError('Contextual retrieval must be on or off.')
    from rag.server import _apply_settings
    arguments = {key: value for key, value in changes.items()
                 if key not in {'revision', 'embedding_source', 'embedding_runtime'}}
    if embedding_choice and source == 'local_runtime':
        arguments['embedding_runtime'] = runtime
    result = _apply_settings(arguments,
                             expected_revision=expected_revision)
    if 'error' in result:
        raise ValueError('Embedding model verification failed; nothing was saved.' if embedding_choice
                         else 'Engine settings were rejected; no changes were saved.')
    return read_config()


def capabilities():
    config = read_config()
    embedding = settings.embedding_config()
    retrieval = config['retrieval']
    from rag.openrouter_pipeline import openrouter_key
    try:
        openrouter_key(embedding)
        hosted = 'unknown'  # A configured key is not a successful provider request.
    except ValueError:
        hosted = 'unavailable'
    def stage(name, provider, model, availability, credential=None):
        return {'stage': name, 'provider': provider, 'model': model, 'display_name': model or provider,
                'location': 'api' if provider in {'openrouter', 'openai_compatible'} else 'local',
                'availability': availability, 'required_credential': credential,
                'context_limit': None, 'dimensions': None, 'discovery_source': 'engine_configuration'}
    embedding_provider = embedding['embed_provider']
    embedding_credential = None
    embedding_location = None
    if embedding_provider == 'openai_compatible':
        host = urlsplit(embedding['embed_base_url']).hostname
        keys = {'api.openai.com': ('openai_key', ('OPENAI_API_KEY',)),
                'generativelanguage.googleapis.com': ('gemini_key', ('GEMINI_API_KEY', 'GOOGLE_API_KEY')),
                'openrouter.ai': ('openrouter_key', ('OPENROUTER_API_KEY',))}
        if host in keys:
            embedding_credential, names = keys[host]
            embedding_availability = ('unknown' if embedding['embed_api_key'].strip()
                                      or any(os.environ.get(name, '').strip() for name in names) else 'unavailable')
        else:
            embedding_availability = 'unknown'
        embedding_location = 'local' if host in {'localhost', '127.0.0.1', '::1'} else 'api'
    elif embedding_provider == 'none' or embedding_provider not in settings.PROVIDERS:
        embedding_availability = 'unavailable'
    else:
        embedding_availability = 'unknown'
    reranker = retrieval['reranker_type']
    stages = [
        stage('lexical', 'tantivy', 'BM25', 'available'),
        stage('embedding', embedding_provider, embedding['embed_model'],
              embedding_availability, embedding_credential),
        stage('reranking', reranker, '' if reranker == 'off' else retrieval['reranker_model'],
              hosted if reranker == 'openrouter' else 'unavailable' if reranker == 'off' else 'unknown',
              'openrouter_key' if reranker == 'openrouter' else None),
        stage('generation', 'openrouter', retrieval['answer_model'], hosted, 'openrouter_key'),
    ]
    if embedding_location is not None:
        stages[1]['location'] = embedding_location
    return {'schema_version': 1, 'stages': stages}


def model_inventory():
    """Discover a bounded local catalog only when a client explicitly asks."""
    from rag import detect, model_catalog

    with settings.frozen_settings():
        configured = capabilities()['stages']
        embedding = settings.embedding_config()
    models = {}
    sources = []

    def add(stage, provider, model, location, source, runtime=None, configured=False):
        if (not isinstance(model, str) or not model.strip() or len(model) > 200
                or any(ord(char) < 32 for char in model)):
            return
        key = (stage, provider, model)
        previous = models.get(key)
        models[key] = {'stage': stage, 'provider': provider, 'model': model,
                       'location': location, 'discovery_source': source,
                       'runtime': runtime, 'configured': configured or bool(previous and previous['configured'])}

    for item in configured:
        if item['stage'] != 'lexical' and item['model'] and item['provider'] not in {'off', 'none'}:
            add(item['stage'], item['provider'], item['model'], item['location'], 'engine_configuration',
                configured=True)

    def collect(source, callback):
        try:
            values, status = callback()
            sources.append({'source': source, 'status': status})
            for entry in values[:20]:
                add(*entry)
        except Exception:
            sources.append({'source': source, 'status': 'error'})

    def ollama():
        result = detect.detect_ollama(embedding['ollama_url'], max_models=20)
        values = [('embedding', 'ollama', model, 'local', 'ollama', 'Ollama')
                  for model in result.get('embedding_models', [])]
        status = ('unavailable' if not result.get('running') else 'truncated' if result.get('truncated')
                  or len(values) > 20 else 'complete')
        return values, status

    def local_runtime():
        values = [('embedding', 'openai_compatible', model, 'local', 'local_runtime', runtime['label'])
                  for runtime in detect.detect_local_runtimes()
                  for model in runtime.get('embedding_models', [])]
        return values, 'truncated' if len(values) > 20 else 'complete'

    def sentence_transformers():
        values = [('embedding', 'sentencetransformer', item['label'], 'local',
                   'sentence_transformers_cache', 'sentence-transformers')
                  for item in detect.detect_cached_models(os.environ)]
        return values, 'truncated' if len(values) > 20 else 'complete'

    def fastembed():
        values = [('embedding', 'fastembed', model, 'local', 'fastembed_cache', 'FastEmbed')
                  for model in model_catalog.cached_cpu_models(os.environ)]
        return values, 'truncated' if len(values) > 20 else 'complete'

    collect('ollama', ollama)
    collect('local_runtime', local_runtime)
    collect('sentence_transformers_cache', sentence_transformers)
    collect('fastembed_cache', fastembed)
    return {'schema_version': 1, 'models': list(models.values()), 'sources': sources}
