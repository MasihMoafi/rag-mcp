import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import multiprocessing

import pytest

from rag.index_lifecycle import active_directory, open_or_build, RebuildRequired


def test_empty_scope_builds_once_without_initializing_models_for_an_empty_index(tmp_path):
    builds = []
    def build(config):
        builds.append(config)
        if not config.get('force_reindex'):
            raise RebuildRequired('empty')
        return SimpleNamespace(config=config)
    instance = open_or_build({'persist_dir': str(tmp_path)}, build)
    assert len(builds) == 1
    assert instance.config['force_reindex'] is True


@pytest.mark.parametrize('generation', [None, {}, '../generation-outside', '/generation-outside', 'missing', 'generation-missing'])
def test_invalid_pointer_does_not_silently_open_or_rebuild_another_directory(tmp_path, generation):
    (tmp_path / 'active-index.json').write_text(json.dumps(generation))
    with pytest.raises(ValueError, match='index generation'):
        active_directory(tmp_path)


def test_legacy_scope_without_pointer_is_still_readable(tmp_path):
    assert active_directory(tmp_path) == tmp_path


def test_active_generation_cannot_be_a_symlink(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (tmp_path / 'generation-link').symlink_to(outside, target_is_directory=True)
    (tmp_path / 'active-index.json').write_text('"generation-link"')
    with pytest.raises(ValueError, match='index generation'):
        active_directory(tmp_path)


def test_slow_refresh_cannot_overwrite_a_completed_competing_refresh(tmp_path):
    started, resume = Event(), Event()
    config = {'persist_dir': str(tmp_path), 'force_reindex': True}
    def slow_build(config):
        started.set()
        assert resume.wait(10), 'test did not release slow builder'
        return SimpleNamespace(config=config)
    with ThreadPoolExecutor(max_workers=1) as workers:
        slow = workers.submit(open_or_build, config, slow_build)
        try:
            assert started.wait(10)
            winner = open_or_build(config, lambda config: SimpleNamespace(config=config))
        finally:
            resume.set()
        with pytest.raises(RuntimeError, match='another refresh'):
            slow.result(timeout=10)
    assert str(active_directory(tmp_path)) == winner.config['persist_dir']
    assert len(list(tmp_path.glob('generation-*'))) == 1


def _competing_process(config, started, resume, outcome):
    def build(config):
        started.set()
        if not resume.wait(10):
            raise RuntimeError('test release timed out')
        return SimpleNamespace(config=config)
    try:
        open_or_build(config, build)
        outcome.put('published')
    except RuntimeError as error:
        outcome.put(str(error))


def test_competing_process_cannot_replace_newer_publication(tmp_path):
    context = multiprocessing.get_context('spawn')
    started, resume, outcome = context.Event(), context.Event(), context.Queue()
    config = {'persist_dir': str(tmp_path), 'force_reindex': True}
    process = context.Process(target=_competing_process, args=(config, started, resume, outcome))
    process.start()
    try:
        assert started.wait(10)
        winner = open_or_build(config, lambda config: SimpleNamespace(config=config))
        resume.set()
        assert 'another refresh' in outcome.get(timeout=10)
        process.join(timeout=10)
        assert process.exitcode == 0
        assert str(active_directory(tmp_path)) == winner.config['persist_dir']
    finally:
        resume.set()
        if process.is_alive():
            process.terminate()
        process.join(timeout=10)
        outcome.close()
