import json
from types import SimpleNamespace

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
