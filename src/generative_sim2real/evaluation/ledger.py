"""Hash-chained C2 campaign ledger with out-of-band seals: a one-shot guard that is tamper-EVIDENT.

State lives in one fixed directory (``outputs/c2_campaign``):

* ``ledger.jsonl``     hash-chained ``start``/``complete``/``abort`` events, replaced atomically under a lock;
* ``.seals/<hash>``    per-registry marker created with O_EXCL after the first ``start`` is ledgered;
* ``.seals/<hash>.complete``  campaign seal written after ``complete`` (records the ledger line it must match);
* ``<hash12>*/``       result directories.

``audit`` cross-checks the three: a seal or result directory whose registry has no ledger start, a missing
ledger, a sealed ``complete`` that the ledger no longer shows, or a broken hash chain all raise. There is no git
history and ``outputs/`` is writable, so someone who rewrites the ledger, the seals and the results together
cannot be detected: this makes silent re-use evident, not impossible.
"""
from __future__ import annotations
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re

LEDGER_NAME = 'ledger.jsonl'
SEAL_DIR = '.seals'
LOCK_NAME = '.lock'


class LedgerError(RuntimeError):
    pass


def _line_hash(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def ledger_path(directory) -> Path:
    return Path(directory) / LEDGER_NAME


def seal_dir(directory) -> Path:
    return Path(directory) / SEAL_DIR


@contextlib.contextmanager
def locked(directory):
    """Exclusive, non-blocking campaign lock: a concurrent launch is refused rather than queued."""
    seal_dir(directory).mkdir(parents=True, exist_ok=True)
    stream = open(seal_dir(directory) / LOCK_NAME, 'a+')
    try:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise LedgerError('Another launch holds the C2 campaign lock; refusing a concurrent start') from error
        yield
    finally:
        stream.close()


def read_ledger(directory) -> list:
    path = ledger_path(directory)
    if not path.exists():
        return []
    entries, previous = [], None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        entry = json.loads(line)
        if entry.get('prev_sha256') != previous or entry.get('index') != number:
            raise LedgerError(f'Ledger {path} chain is broken at line {number}; it was edited or truncated')
        previous = _line_hash(line)
        entries.append(entry)
    return entries


def line_hashes(directory) -> list:
    path = ledger_path(directory)
    return [_line_hash(line) for line in path.read_text().splitlines()] if path.exists() else []


def append(directory, event: dict, *, held=False) -> dict:
    """Atomic append: rewrite the file through a temporary and os.replace, under the campaign lock."""
    with contextlib.nullcontext() if held else locked(directory):
        path = ledger_path(directory)
        entries = read_ledger(directory)
        lines = path.read_text().splitlines() if entries else []
        previous = _line_hash(lines[-1]) if lines else None
        entry = {'index': len(entries) + 1,
                 'utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                 'host': platform.node(), 'pid': os.getpid(), **event, 'prev_sha256': previous}
        lines.append(json.dumps(entry, sort_keys=True, allow_nan=False))
        temporary = path.with_name(f'{path.name}.tmp{os.getpid()}')
        try:
            with open(temporary, 'w') as stream:
                stream.write('\n'.join(lines) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return entry


def write_marker(directory, registry_sha256: str, info: dict):
    """O_EXCL per-registry marker, created once after the first ledgered start of that registry."""
    path = seal_dir(directory) / registry_sha256
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as error:
        raise LedgerError(f'Marker for registry {registry_sha256[:12]} already exists without a ledger start') from error
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps({'registry_sha256': registry_sha256, **info}, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_campaign_seal(directory, registry_sha256: str, entry: dict, line_sha256: str, output_digest: str):
    path = seal_dir(directory) / f'{registry_sha256}.complete'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps({'registry_sha256': registry_sha256, 'ledger_complete_index': entry['index'],
                                 'ledger_complete_line_sha256': line_sha256, 'output_files_digest': output_digest},
                                sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


_HASH = re.compile(r'^[0-9a-f]{64}$')
_RESULT_DIR = re.compile(r'^([0-9a-f]{12})(\..*)?$')


def audit(directory) -> dict:
    """Cross-check ledger, seals and result directories; raise LedgerError on any inconsistency."""
    directory = Path(directory)
    entries = read_ledger(directory)
    hashes = line_hashes(directory)
    starts = {e['registry_sha256'] for e in entries if e['event'] == 'start'}
    seals = seal_dir(directory)
    problems = []
    if seals.is_dir():
        for p in sorted(seals.iterdir()):
            if _HASH.match(p.name):
                if p.name not in starts:
                    problems.append(f'seal {p.name[:12]} has no ledger start (ledger truncated, replaced or deleted)')
            elif p.name.endswith('.complete') and _HASH.match(p.name[:-9]):
                seal = json.loads(p.read_text())
                index = seal['ledger_complete_index']
                ok = (index <= len(entries) and entries[index - 1]['event'] == 'complete'
                      and entries[index - 1]['registry_sha256'] == seal['registry_sha256']
                      and hashes[index - 1] == seal['ledger_complete_line_sha256'])
                if not ok:
                    problems.append(f'campaign seal {p.name[:12]} is not matched by the ledger complete event')
    if directory.is_dir():
        for p in sorted(directory.iterdir()):
            match = _RESULT_DIR.match(p.name)
            if p.is_dir() and match and not any(h.startswith(match.group(1)) for h in starts):
                problems.append(f'result directory {p.name} has no ledger start')
    if problems:
        raise LedgerError('C2 campaign state is inconsistent: ' + '; '.join(problems))
    return {'entries': entries, 'starts': starts}


def _alive(entry) -> bool:
    if entry.get('host') != platform.node():
        return True                              # cannot prove a remote process is gone
    try:
        os.kill(entry['pid'], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def plan_start(entries: list, registry_sha256: str, disclosure=None) -> dict:
    """What a campaign start would mean for this registry; raises LedgerError when it must be refused."""
    starts = [e for e in entries if e['event'] == 'start']
    previous = []
    for e in starts:
        if e['registry_sha256'] not in previous:
            previous.append(e['registry_sha256'])
    if any(e['event'] == 'complete' and e['registry_sha256'] == registry_sha256 for e in entries):
        raise LedgerError('This registry already completed a C2 campaign; its results exist and are not overwritten')
    orphans = []
    for sha in previous:
        mine = [e for e in entries if e['registry_sha256'] == sha]
        if mine[-1]['event'] == 'start':
            if _alive(mine[-1]):
                raise LedgerError(f'A C2 campaign for registry {sha[:12]} is still running (pid {mine[-1]["pid"]} on '
                                  f'{mine[-1]["host"]}); refusing a concurrent start')
            orphans.append(sha)
    others = [h for h in previous if h != registry_sha256]
    if registry_sha256 in previous:
        return {'experiment_index': previous.index(registry_sha256) + 1, 'new_disclosed_experiment': False,
                'attempt': sum(e['registry_sha256'] == registry_sha256 for e in starts) + 1,
                'other_registries': others, 'earlier_registries': others, 'disclosure': None, 'orphaned': orphans}
    if previous and not (isinstance(disclosure, str) and disclosure.strip()):
        raise LedgerError(
            'C2 inference was already started under a different registry (' + ', '.join(h[:12] for h in previous)
            + '). A changed protocol is a new disclosed experiment: supply a disclosure reason to proceed.')
    return {'experiment_index': len(previous) + 1, 'new_disclosed_experiment': bool(previous), 'attempt': 1,
            'other_registries': others, 'earlier_registries': others,
            'disclosure': disclosure.strip() if previous else None, 'orphaned': orphans}
