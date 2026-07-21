#!/usr/bin/env python3
"""Durable artifact writes — because every runner in this tree could destroy its own history.

Every runner in `scripts/` ends the same way:

    Path(args.out).write_text(json.dumps({...}) + "\\n", encoding="utf-8")

That single line has two defects and both of them matter for a run that takes hours. It is not
atomic, so a SIGKILL, an OOM or a full disk halfway through leaves a truncated file that is no
longer valid JSON — and it has already overwritten the previous good one, so there is nothing to
fall back to. And it is not durable: `write_text` returns once the bytes are in the page cache, so
a machine that loses power reports success for a file that never reached the platter.

The fix is the usual one, written down once instead of ten times: serialise into a sibling
temporary file, flush it, fsync it, then `os.replace` it over the target. `os.replace` is atomic
within a filesystem, so a reader either sees the whole old file or the whole new one and never a
half-written mixture. The directory itself is fsynced afterwards, because the rename is metadata
and lives in the directory's own journal entry — without that fsync the rename can be lost even
though the file contents survived.

None of this is free: an fsync costs a disk round trip. That is the right trade for an artifact
written once at the end of a stage, and the wrong trade for a per-row log, which is why the ledger
in `ledger.py` batches differently.

    from artifact_io import write_json_atomic, sha256_of
    write_json_atomic(out_path, payload)
    manifest["artifact_sha256"][out_path.name] = sha256_of(out_path)
"""
import hashlib
import json
import os
from pathlib import Path

CHUNK = 1 << 20


def write_json_atomic(path, obj, indent=1):
    """Serialise `obj` to `path` so that a crash can never leave the target unreadable.

    Returns the sha256 of exactly the bytes written, so a caller building a run manifest does not
    have to read the file back and hope nothing touched it in between.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(obj, indent=indent, ensure_ascii=False) + "\n").encode("utf-8")

    # The temporary file is a sibling, not /tmp: os.replace is only atomic within one filesystem,
    # and on this machine /tmp may well be a different one.
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def _fsync_dir(d):
    """Persist the rename itself. Without this the new contents can survive while the directory
    entry that points at them does not, which is the failure mode that looks like the write simply
    never happened."""
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass                       # some filesystems refuse directory fsync; the replace still ran
    finally:
        os.close(fd)


def sha256_of(path):
    """Streamed, so a 178 MB store does not have to fit in memory to be hashed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, default=None):
    """Read an artifact, treating a truncated file as absent rather than as a crash.

    A pre-atomic-write artifact may still be sitting on disk in a half-written state. Returning the
    default lets a resuming stage decide to recompute rather than die on a JSONDecodeError far from
    the actual cause.
    """
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default
