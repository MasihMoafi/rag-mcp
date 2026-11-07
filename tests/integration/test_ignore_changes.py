import subprocess

from rag import server


def test_ignore_changes_take_effect_in_a_running_server(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    document = tmp_path / 'document.md'
    document.write_text('A searchable document.\n')
    rules = tmp_path / '.gitignore'
    rules.write_text('other.md\n')
    assert str(document) in server._validate_rag_scope(tmp_path).files
    rules.write_text('document.md\n')
    assert str(document) not in server._validate_rag_scope(tmp_path).files
    rules.write_text('other.md\n')
    assert str(document) in server._validate_rag_scope(tmp_path).files
