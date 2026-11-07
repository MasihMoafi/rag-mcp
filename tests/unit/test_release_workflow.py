"""Publication must wait for quality and tests of the artifacts being published."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def publication_has_gates(jobs, publication):
    seen = set()

    def visit(job):
        if job in seen:
            return
        seen.add(job)
        needs = jobs[job].get('needs', [])
        for dependency in [needs] if isinstance(needs, str) else needs:
            visit(dependency)

    visit(publication)
    return {'build', 'quality', 'installed-artifact'} <= seen


def test_both_publications_depend_on_quality_and_installed_artifacts():
    release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
    quality = yaml.safe_load((ROOT / '.github/workflows/python-quality.yml').read_text())
    jobs = release['jobs']
    assert {'keyless', 'lint', 'types'} <= quality['jobs'].keys()
    assert set(quality['jobs']['keyless']['strategy']['matrix']['python']) == {'3.10', '3.13'}
    assert jobs['quality']['uses'] == './.github/workflows/python-quality.yml'
    for name in ('publish-pypi', 'publish-npm'):
        assert publication_has_gates(jobs, name)
    steps = jobs['installed-artifact']['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert 'sha256sum -c SHA256SUMS' in commands
    assert 'tests/install/smoke.sh' in commands
    assert any(step.get('with', {}).get('name') == 'release-artifacts' for step in steps)
    weakened = dict(jobs)
    weakened['publish-pypi'] = {'needs': ['build']}
    assert not publication_has_gates(weakened, 'publish-pypi')
    assert not publication_has_gates(weakened, 'publish-npm')


def test_manual_release_rehearsal_runs_checks_but_never_publishes():
    release = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
    triggers = release.get('on', release.get(True))
    assert 'workflow_dispatch' in triggers
    jobs = release['jobs']
    for name in ('quality', 'build', 'installed-artifact'):
        assert "github.event_name == 'workflow_dispatch'" in jobs[name]['if']
    for name in ('publish-pypi', 'publish-npm'):
        condition = jobs[name]['if']
        assert "github.event_name == 'push'" in condition
        assert "vars.PUBLISH_ENABLED == 'true'" in condition
    build_commands = '\n'.join(step.get('run', '') for step in jobs['build']['steps'])
    assert 'check_release_artifacts.py' in build_commands
    assert 'workflow_dispatch' in build_commands


def test_install_smoke_searches_after_provider_setup():
    smoke = (ROOT / 'tests/install/smoke.sh').read_text()
    after_setup = smoke.split('setup --provider openai_compatible', 1)[1]
    assert '"$rag" search' in after_setup
    assert '"passages"' in after_setup
    assert '--skip-fastembed' not in (ROOT / 'tests/install/run.sh').read_text()
