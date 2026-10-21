"""Small durable JSON cache shared by web and reverse-image retrieval."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


def canonical_json(value: Any) -> str:
    """Stable JSON without lossy conversions or non-standard NaN values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


class EvidenceCache:
    """Atomic persistent cache; one instance may be shared across worker threads.

    Returned values are independent decoded dictionaries. Reads never update the
    database, so an offline cache lookup does not alter a captured evidence set.
    WAL and SQLite's busy timeout also permit multiple local worker processes.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser().resolve())
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=30, check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS evidence_cache (
                    namespace TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key_hash)
                )
            """)
        self._reads = 0
        self._hits = 0
        self._writes = 0

    @staticmethod
    def key_hash(key: dict) -> str:
        if not isinstance(key, dict):
            raise TypeError("Cache keys must be dictionaries")
        return hashlib.sha256(canonical_json(key).encode("utf-8")).hexdigest()

    @staticmethod
    def _namespace(namespace: str) -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("Cache namespace must be a non-empty string")

    def get(self, namespace: str, key: dict) -> dict | None:
        self._namespace(namespace)
        digest = self.key_hash(key)
        with self._lock:
            self._reads += 1
            row = self._connection.execute(
                "SELECT value_json FROM evidence_cache WHERE namespace=? AND key_hash=?",
                (namespace, digest),
            ).fetchone()
            if row is None:
                return None
            self._hits += 1
        return json.loads(row[0])

    def put(self, namespace: str, key: dict, value: dict) -> None:
        self._namespace(namespace)
        digest = self.key_hash(key)
        if not isinstance(value, dict):
            raise TypeError("Cache values must be dictionaries")
        key_json = canonical_json(key)
        value_json = canonical_json(value)
        status = value.get("status", "unspecified")
        if not isinstance(status, str):
            status = "unspecified"
        now = time.time()
        with self._lock, self._connection:
            self._connection.execute("""
                INSERT INTO evidence_cache
                    (namespace, key_hash, key_json, value_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (namespace, key_hash) DO UPDATE SET
                    key_json=excluded.key_json,
                    value_json=excluded.value_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
            """, (namespace, digest, key_json, value_json, status, now, now))
            self._writes += 1

    def stats(self) -> dict:
        """Report durable entry counts and this instance's lookup counters."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT namespace, status, COUNT(*) FROM evidence_cache GROUP BY namespace, status"
            ).fetchall()
            namespaces: Counter = Counter()
            statuses: Counter = Counter()
            for namespace, status, count in rows:
                namespaces[namespace] += count
                statuses[status] += count
            return {
                "entries": sum(namespaces.values()),
                "by_namespace": dict(namespaces),
                "by_status": dict(statuses),
                "reads": self._reads,
                "hits": self._hits,
                "misses": self._reads - self._hits,
                "writes": self._writes,
            }

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> EvidenceCache:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
