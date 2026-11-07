"""Core of every paid batch run under evals/ (the guards from skills/dev/AGENTS.md, requirements 1, 2 and 8).
A runner gives it the input sets, one `call(set_name, item)` that makes one paid call, a `proves(result)`
that reads the field proving success (for a search: `passages`), the unit price and the cap. `start()` then:
  1. smoke-tests `smoke` items from every set, prints each output, and stops at the first unproven result;
  2. prints the cost estimate, probe cost times the remaining items, and waits for "yes";
  3. runs the rest, stops at the first unproven result, and stops before a call that would pass the cap, counted from
     its own calls, never from the key's reported usage (it lags). A resumed run redoes every item whose record lacks
     the proving field: "it parsed" is not done.
`isolated_env()` refuses to start unless every store rag-mcp writes is outside the real store and outside a git-ignored
folder of the repo, and takes one lock per store so two runs never share one."""
import fcntl, json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORES = ('RAG_MCP_CONFIG', 'RAG_MCP_DATA_DIR', 'RAG_MCP_WORKSPACE_DIR', 'RAG_MCP_TRACE_DIR')
_locks = []  # held for the life of the process


class Stop(SystemExit):
    """The run stopped on purpose; the message says where and why."""


def real_stores(environ):
    """Where rag-mcp writes when the four variables are unset (rag/settings.py, index_config.py, workspaces.py, tracing.py)."""
    home = Path(environ.get('HOME') or Path.home())
    return {Path(environ.get('XDG_CONFIG_HOME') or home / '.config') / 'rag',
            Path(environ.get('XDG_DATA_HOME') or home / '.local/share') / 'rag' / 'index',
            Path(environ.get('XDG_STATE_HOME') or home / '.local/state') / 'ragmcp' / 'workspaces',
            Path(environ.get('XDG_STATE_HOME') or home / '.local/state') / 'ragmcp' / 'traces'}


def _ignored(repo, path):
    try:
        path.relative_to(repo)
    except ValueError:
        return False
    return subprocess.run(['git', '-C', str(repo), 'check-ignore', '-q', str(path)], capture_output=True).returncode == 0


def isolated_env(run, environ, repo=ROOT):
    """The run's own config, index, workspaces and traces under `run` (use ~/.cache/rag-experiments/<run>/), each
    checked against the real store and the repo's ignored folders and locked; returns the environment to pass to rag."""
    run = Path(run).expanduser().resolve()
    paths = dict(zip(STORES, (run / 'config.json', run / 'index', run / 'workspaces', run / 'traces')))
    real = real_stores(environ)
    for name, path in paths.items():
        if any(path == r or r in path.parents or path in r.parents for r in real):
            raise Stop(f'{name}={path} is the real store; use ~/.cache/rag-experiments/<run>/')
        if _ignored(repo, path):
            raise Stop(f'{name}={path} is inside a git-ignored folder of the repo, where rag sees no documents; use ~/.cache/rag-experiments/<run>/')
    for name, path in paths.items():
        if name == 'RAG_MCP_CONFIG':
            continue
        path.mkdir(parents=True, exist_ok=True)
        handle = open(path / '.paidrun.lock', 'w')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Stop(f'{name}={path}: another run holds this store; give each run its own') from None
        _locks.append(handle)
    return {**environ, **{name: str(path) for name, path in paths.items()}}


def cli_search(python, code, env, workspace, question, waits=120):
    """One `rag search` through the CLI; waits out another query's turn at the engine instead of recording it as a result."""
    for _ in range(waits):
        run = subprocess.run([str(python), '-m', 'rag.cli', 'search', '--workspace', workspace, '--json', question],
                             cwd=code, env=env, capture_output=True, text=True, timeout=3600)
        try:
            result = json.loads(run.stdout)
        except ValueError:
            return {'error': (run.stderr or run.stdout)[-1500:], 'code': run.returncode}
        if result.get('code') != 'engine_busy':
            return result
        time.sleep(5)
    return {'error': f'engine busy for {waits * 5} seconds'}


def has_passages(result):
    return bool(result.get('passages'))


class Run:
    def __init__(self, sets, call, proves, unit, cap, out, smoke=5, usage=None, confirm=input, log=print):
        self.sets, self.call, self.proves, self.unit, self.cap = sets, call, proves, unit, cap
        self.out, self.smoke_count, self.usage, self.confirm, self.log = Path(out), smoke, usage, confirm, log
        self.calls = 0

    def done(self):
        if not self.out.exists():
            return set()
        return {(r['collection'], r['n']) for r in map(json.loads, self.out.open()) if 'result' in r and self.proves(r['result'])}

    def todo(self):
        done = self.done()
        return [(name, i, item) for name, items in self.sets.items() for i, item in enumerate(items, 1) if (name, i) not in done]

    def one(self, name, i, item):
        if (self.calls + 1) * self.unit > self.cap:
            raise Stop(f'stopped before {name} #{i}: {self.calls} calls made, one more passes the ${self.cap} cap')
        began = time.monotonic()
        result = self.call(name, item)
        self.calls += 1
        with self.out.open('a') as out:
            out.write(json.dumps({'collection': name, 'n': i, 'row': item, 'seconds': round(time.monotonic() - began, 2), 'result': result},
                                 ensure_ascii=False) + '\n')
        if not self.proves(result):
            raise Stop(f'stopped at the first failure ({name} #{i}): {json.dumps(result, ensure_ascii=False)[:300]}')
        return result

    def smoke(self):
        """About `smoke` items from every set; every output printed for reading, the first failure stops the run."""
        before = self.usage() if self.usage else 0.0
        probes = 0
        for name in self.sets:
            for _, i, item in [t for t in self.todo() if t[0] == name][:self.smoke_count]:
                result = self.one(name, i, item)
                self.log(f'smoke {name} #{i}: {json.dumps(result, ensure_ascii=False)[:300]}')
                probes += 1
        spent = (self.usage() - before) if self.usage else probes * self.unit
        return probes, spent

    def start(self):
        probes, spent = self.smoke()
        remaining = len(self.todo())
        per = spent / probes if probes else self.unit
        self.log(f'estimate: {probes} probes cost ${spent:.4f}, ${per:.4f} per item x {remaining} remaining = ${per * remaining:.4f}; cap ${self.cap} ({self.unit} per call by own count)')
        if self.confirm('run the rest? type yes: ').strip() != 'yes':
            raise Stop('not confirmed; nothing more spent')
        for name, i, item in self.todo():
            self.one(name, i, item)
        return self.calls
