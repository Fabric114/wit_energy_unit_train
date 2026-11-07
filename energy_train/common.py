"""Small filesystem helpers shared by the offline tools (Linux)."""

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def valid_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("session/group_id 仅允许字母、数字、下划线和短横线，长度 1～96")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def atomic_json(path, value):
    path = Path(path)
    text = json_text(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"目录正在被另一个进程使用 (locked): {path.parent}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextlib.contextmanager
def new_output_directory(path):
    """Publish only a complete export; never merge with an existing directory."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.parent / f".{path.name}.publish.lock"):
        if path.exists():
            raise FileExistsError(f"输出已存在，请使用新目录: {path}")
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        try:
            yield staging
            if path.exists():
                raise FileExistsError(f"输出已存在: {path}")
            staging.rename(path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
