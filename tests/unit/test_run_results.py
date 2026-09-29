import pytest

from rag.run_results import RunResultStore


@pytest.mark.parametrize('data', ['[]', '{broken', 'null'])
def test_corrupt_record_is_storage_error(tmp_path, data):
    store = RunResultStore(tmp_path)
    (tmp_path / ('a' * 32 + '.json')).write_text(data)
    with pytest.raises(OSError):
        store.read('a' * 32)


def test_save_does_not_follow_predictable_temporary_symlink(tmp_path):
    store = RunResultStore(tmp_path / 'results')
    store.root.mkdir()
    target = tmp_path / 'untouched'
    target.write_text('preserve')
    (store.root / ('a' * 32 + '.tmp')).symlink_to(target)
    record = {'schema_version': 1, 'run_id': 'a' * 32, 'status': 'success'}
    store.save(record)
    assert target.read_text() == 'preserve'
    assert store.read('a' * 32) == record
    assert (store.root / ('a' * 32 + '.json')).stat().st_mode & 0o777 == 0o600
