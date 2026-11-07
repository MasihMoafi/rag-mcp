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
