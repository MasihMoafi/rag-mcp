from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag.core import RAGPipelineV2


def test_contextualization_failure_keeps_evidence_without_logging_private_payload(capsys):
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {'use_contextual': True}
    pipeline.contextualizer = Mock(side_effect=RuntimeError('private-provider-payload'))
    assert pipeline._add_context_to_chunk('Original evidence.', 'Document context') == 'Original evidence.'
    output = capsys.readouterr()
    assert 'RuntimeError' in output.err
    assert 'private-provider-payload' not in output.out + output.err


def test_directory_formats_case_and_explicit_paths_are_preserved(tmp_path):
    extensions = (
        '.rs', '.toml', '.yaml', '.yml', '.py', '.md', '.txt', '.json',
        '.js', '.ts', '.tsx', '.jsx', '.c', '.h', '.cpp', '.hpp',
        '.go', '.sh', '.bash', '.zsh', '.css', '.html', '.sql', '.java',
        '.kt', '.proto', '.pdf', '.ipynb', '.rst', '.ini', '.cfg', '.conf',
        '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp', '.docx', '.pptx', '.xlsx', '.csv'
    )
    directory = tmp_path / "directory"
    directory.mkdir()
    supported = [directory / f"source{ext.upper()}" for ext in extensions]
    for path in supported:
        path.touch()
    (directory / "excluded.exe").touch()
    nested = directory / "nested"
    nested.mkdir()
    (nested / "inside.txt").touch()
    explicit = tmp_path / "explicit.unknown"
    explicit.touch()
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"document_paths": [str(directory), str(explicit), str(tmp_path / "missing")],
                       "chunk_size": 100, "chunk_overlap": 0}
    pipeline._extract_content = Mock(return_value="Source evidence.")
    pipeline.vector_db = SimpleNamespace(add_documents_batch=Mock())
    pipeline.bm25_index = None
    pipeline._save_database = Mock()

    pipeline._index_documents()

    expected = {str(path) for path in supported + [explicit]}
    assert {call.args[0] for call in pipeline._extract_content.call_args_list} == expected
    docs = pipeline.vector_db.add_documents_batch.call_args.args[0]
    assert {doc["file_path"] for doc in docs} == expected
    assert len(docs) == len(expected)
    assert all(doc["content"] == doc["original_content"] == "Source evidence." for doc in docs)
    pipeline.vector_db.add_documents_batch.assert_called_once_with(docs, batch_size=512)
    pipeline._save_database.assert_called_once_with(docs)


@pytest.mark.parametrize("contextual", [False, True])
def test_form_feed_cleanup_preserves_citation_and_context_input(tmp_path, contextual):
    path = tmp_path / "pages.pdf"
    path.touch()
    pipeline = RAGPipelineV2.__new__(RAGPipelineV2)
    pipeline.config = {"document_paths": [str(path)], "chunk_size": 100,
                       "chunk_overlap": 0, "use_contextual": contextual}
    pipeline._extract_content = Mock(return_value="First page\fSecond page")
    pipeline.contextualizer = Mock(return_value="Context")
    pipeline.vector_db = SimpleNamespace(add_documents_batch=Mock())
    pipeline.bm25_index = None
    pipeline._save_database = Mock()

    pipeline._index_documents()

    doc = pipeline.vector_db.add_documents_batch.call_args.args[0][0]
    assert doc["original_content"] == "First page Second page"
    assert doc["content"] == ("Context\n" if contextual else "") + "First page Second page"
    assert (doc["location"], doc["page_number"]) == ("p.1-2", 1)
    if contextual:
        pipeline.contextualizer.assert_called_once_with("First page Second page", "First page\fSecond page")
    else:
        pipeline.contextualizer.assert_not_called()
