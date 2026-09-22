"""Gate arithmetic and rejection paths, not a model-quality evaluation."""
import hashlib
import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location('gate', Path(__file__).resolve().parents[2] / 'evals/check_release.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def bundle():
    cases = []
    for stratum in sorted(gate.STRATA):
        for index in range(10):
            negative = stratum == 'unsupported'
            claims = [] if negative else ['claim']
            cases.append(dict(id=f'{stratum}-{index}', family='unseen-family', stratum=stratum,
                              reviewed=True, scope_safe=True, injection_safe=True, public_boundary_safe=True,
                              status='success', material_invention=False, required_claims=claims,
                              candidate_claims=claims, selected_claims=claims, answered_claims=claims,
                              output_claims_supported=[] if negative else [True],
                              answerability='unsupported' if negative else 'answerable', abstained=negative))
    return dict(revision='revision', config_sha256='config', split='heldout', reviewer='reviewer',
                author='author', frozen_before_run=True, development_families=['previous-book'], cases=cases)


def test_complete_reviewed_bundle_passes(bundle):
    assert gate.check(bundle, 'revision', 'config')['passed']


@pytest.mark.parametrize('field,value', [('revision', 'stale'), ('config_sha256', 'stale'),
    ('split', 'development'), ('reviewer', 'author'), ('frozen_before_run', False), ('development_families', [])])
def test_untrusted_or_stale_bundle_fails(bundle, field, value):
    bundle[field] = value
    assert not gate.check(bundle, 'revision', 'config')['passed']


@pytest.mark.parametrize('field,value', [('family', 'previous-book'), ('reviewed', False),
    ('scope_safe', False), ('injection_safe', False), ('public_boundary_safe', False),
    ('material_invention', True), ('status', 'error'), ('candidate_claims', ['invented-id'])])
def test_one_unsafe_or_invalid_case_blocks_release(bundle, field, value):
    bundle['cases'][0][field] = value
    assert not gate.check(bundle, 'revision', 'config')['passed']


def test_missing_case_cannot_shrink_denominator(bundle):
    bundle['cases'].pop()
    assert not gate.check(bundle, 'revision', 'config')['passed']


def test_failed_stratum_is_not_hidden_by_overall_average(bundle):
    for case in bundle['cases'][:3]:
        case['answered_claims'] = []
    result = gate.check(bundle, 'revision', 'config')
    assert result['metrics']['complete_answers'] == .94
    assert not result['passed']
    assert any('below 80%' in p for p in result['problems'])


def test_abstention_is_not_a_complete_partial_answer(bundle):
    for case in bundle['cases'][:10]:
        case.update(answerability='partial', limits_stated=False)
    assert not gate.check(bundle, 'revision', 'config')['passed']


def test_cli_binds_judgments_to_unchanged_artifacts(bundle, tmp_path):
    artifact = tmp_path / 'evidence.txt'
    artifact.write_text('controlled test artifact')
    reference = {'path': artifact.name, 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    for case in bundle['cases']:
        case.update(source=reference, trace=reference, answer=reference)
    path = tmp_path / 'bundle.json'
    path.write_text(json.dumps(bundle))
    command = [sys.executable, gate.__file__, str(path), '--revision', 'revision', '--config-sha256', 'config']
    assert subprocess.run(command, capture_output=True).returncode == 0
    artifact.write_text('changed after review')
    assert subprocess.run(command, capture_output=True).returncode == 1
    path.unlink()
    assert subprocess.run(command, capture_output=True).returncode == 1
