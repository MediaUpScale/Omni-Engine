# -*- coding: utf-8 -*-
"""Resilient Aiwake ↔ local Supermemory bridge.

Production history lives in the ``aiwake_production_history`` container.
Channel rules live in ``aiwake_system_rules``. If localhost:6767 is down,
every call degrades to ``history_seeds.json`` string comparison so the
debate pipeline never crashes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

_LOG = logging.getLogger("aiwake.supermemory")

DEFAULT_BASE_URL = "http://localhost:6767"
HISTORY_CONTAINER = "aiwake_production_history"
RULES_CONTAINER = "aiwake_system_rules"
LEDGER_FILENAME = "history_seeds.json"
SIMILARITY_THRESHOLD = 0.75
HEALTH_PATH = "/v3/health"
HEALTH_TIMEOUT_S = 1.5
REQUEST_TIMEOUT_S = 8.0
OFFLINE_WARNING = (
    "[OmniEngine] Supermemory offline at localhost:6767. "
    "Falling back to local JSON ledger."
)
GOLDEN_RULE = (
    "Rule: Strictly maximum 3 hashtags per post. Captions must use double "
    "line breaks. Topics must strictly reflect actual debate content without "
    "hallucinated defaults."
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_BRIDGE: SupermemoryBridge | None = None
_START_ATTEMPTED = False


def _port_open(host: str = "127.0.0.1", port: int = 6767, *, timeout_s: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def ensure_supermemory_running(
    *,
    host: str = "127.0.0.1",
    port: int = 6767,
    wait_s: float = 8.0,
) -> bool:
    """Ensure the local Supermemory server is listening.

    On Windows an offline server is started silently inside WSL. A process
    makes at most one launch attempt, then polls the TCP port until the
    service is ready. Paths and WSL identity remain environment-overridable
    for installations that differ from the workstation defaults.
    """
    global _START_ATTEMPTED

    # Load project API keys before spawning WSL. Supermemory needs one model
    # provider during startup, and a Windows .env is not automatically
    # visible inside WSL.
    _load_environment()

    if _port_open(host, port):
        return True
    if _START_ATTEMPTED:
        return False
    _START_ATTEMPTED = True

    if sys.platform != "win32":
        _LOG.warning("Supermemory is offline and automatic WSL startup is only available on Windows")
        return False

    distro = (os.getenv("SUPERMEMORY_WSL_DISTRO") or "Ubuntu").strip()
    user = (os.getenv("SUPERMEMORY_WSL_USER") or "freedom_or_death").strip()
    server = (
        os.getenv("SUPERMEMORY_WSL_SERVER")
        or "/home/freedom_or_death/.supermemory/bin/supermemory-server"
    ).strip()
    command = ["wsl", "-d", distro, "-u", user, server]
    child_env = os.environ.copy()
    provider_keys = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    )
    shared_keys = [name for name in provider_keys if child_env.get(name)]
    if shared_keys:
        existing_wslenv = child_env.get("WSLENV", "")
        entries = [entry for entry in existing_wslenv.split(":") if entry]
        entries.extend(name for name in shared_keys if name not in entries)
        child_env["WSLENV"] = ":".join(entries)
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
            close_fds=True,
            env=child_env,
        )
    except OSError as exc:
        _LOG.warning("could not auto-start Supermemory via WSL (%s)", exc)
        return False

    _LOG.info("Supermemory was offline; started it silently via WSL and am waiting for port %d", port)
    deadline = time.monotonic() + max(2.0, wait_s)
    while time.monotonic() < deadline:
        if _port_open(host, port):
            _LOG.info("Supermemory is accepting connections on %s:%d", host, port)
            return True
        time.sleep(0.25)
    _LOG.warning("Supermemory did not become ready within %.1fs", wait_s)
    return False


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_environment() -> None:
    try:
        from channels_config.aiwake.settings import load_environment
    except ImportError:  # pragma: no cover — standalone extraction
        return
    try:
        load_environment()
    except Exception:  # noqa: BLE001 — missing dotenv must not abort a debate
        return


def _default_ledger_path() -> Path:
    try:
        from channels_config.aiwake.settings import resolve_store_dir

        return resolve_store_dir() / LEDGER_FILENAME
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[1] / "store" / LEDGER_FILENAME


def _tokenise(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_similarity(left: str, right: str) -> float:
    """0–1 overlap used by the offline ledger."""
    if _normalise(left) and _normalise(left) == _normalise(right):
        return 1.0
    return _jaccard(_tokenise(left), _tokenise(right))


def _hit_score(hit: Any) -> float:
    if hit is None:
        return 0.0
    if isinstance(hit, dict):
        raw = hit.get("similarity", hit.get("score", hit.get("similarity_score", 0.0)))
    else:
        raw = (
            getattr(hit, "similarity", None)
            or getattr(hit, "score", None)
            or getattr(hit, "similarity_score", 0.0)
        )
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _search_results(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        rows = payload.get("results") or payload.get("memories") or []
        return list(rows) if isinstance(rows, list) else []
    rows = getattr(payload, "results", None)
    if rows is None:
        rows = getattr(payload, "memories", None)
    return list(rows or [])


def session_memory_text(*, session_id: str, topic: str, hook: str, quote: str) -> str:
    return (
        f"Aiwake approved session {session_id}.\n"
        f"Topic: {topic}\n"
        f"Opening hook: {hook}\n"
        f"Core quote: {quote}"
    )


class SupermemoryBridge:
    """Thin client around local Supermemory with a JSON-ledger fallback."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        ledger_path: Path | None = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        force_offline: bool = False,
        health_timeout_s: float = HEALTH_TIMEOUT_S,
    ) -> None:
        _load_environment()
        self.base_url = (base_url or os.getenv("SUPERMEMORY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = (api_key or os.getenv("SUPERMEMORY_API_KEY") or "").strip()
        if api_key is None and self.base_url in {
            "http://localhost:6767",
            "http://127.0.0.1:6767",
        }:
            local_key = Path(__file__).resolve().parents[3] / ".supermemory" / "api-key"
            try:
                self.api_key = local_key.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        self._persist_durable = ledger_path is None
        self.ledger_path = Path(ledger_path) if ledger_path is not None else _default_ledger_path()
        self.similarity_threshold = float(similarity_threshold)
        self.history_container = HISTORY_CONTAINER
        self.rules_container = RULES_CONTAINER
        self._client: Any = None
        self.is_active = False
        if not force_offline and not _env_flag("SUPERMEMORY_DISABLED"):
            if self.base_url in {
                "http://localhost:6767",
                "http://127.0.0.1:6767",
            }:
                ensure_supermemory_running()
            self.is_active = self._ping(health_timeout_s)
        if not self.is_active:
            _LOG.warning(OFFLINE_WARNING)

    def _ping(self, timeout_s: float) -> bool:
        try:
            import requests
        except ImportError:
            return self._ping_sdk(timeout_s)
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.get(
                f"{self.base_url}{HEALTH_PATH}",
                headers=headers,
                timeout=timeout_s,
            )
            if response.status_code >= 400:
                return False
            body = response.json() if response.content else {}
            status = str(body.get("status") or "").strip().lower()
            return status in {"", "ok", "healthy", "ready"}
        except Exception:  # noqa: BLE001 — offline is a valid operating mode
            return False

    def _ping_sdk(self, timeout_s: float) -> bool:
        try:
            import httpx
        except ImportError:
            return False
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = httpx.get(
                f"{self.base_url}{HEALTH_PATH}",
                headers=headers,
                timeout=timeout_s,
            )
            return response.status_code < 400
        except Exception:  # noqa: BLE001
            return False

    def _client_or_none(self) -> Any | None:
        if not self.is_active:
            return None
        if self._client is not None:
            return self._client
        try:
            from supermemory import Supermemory
        except ImportError:
            _LOG.warning("supermemory SDK missing; staying on the local JSON ledger")
            self.is_active = False
            return None
        try:
            self._client = Supermemory(
                api_key=self.api_key or None,
                base_url=self.base_url,
                timeout=REQUEST_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Supermemory client init failed (%s); using local ledger", exc)
            self.is_active = False
            return None
        return self._client

    def _deactivate(self, reason: str) -> None:
        if self.is_active:
            _LOG.warning("Supermemory request failed (%s). Falling back to local JSON ledger.", reason)
        self.is_active = False
        self._client = None

    def _search(self, query: str, *, container_tag: str, search_mode: str) -> list[Any]:
        if self.is_active and self.base_url in {
            "http://localhost:6767",
            "http://127.0.0.1:6767",
        }:
            try:
                import requests

                response = requests.post(
                    f"{self.base_url}/v3/search",
                    headers=(
                        {"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key
                        else {}
                    ),
                    json={
                        "q": query,
                        "container_tag": container_tag,
                        "search_mode": search_mode,
                        "limit": 5,
                    },
                    timeout=REQUEST_TIMEOUT_S,
                )
                response.raise_for_status()
                return _search_results(response.json())
            except Exception as exc:  # noqa: BLE001
                self._deactivate(str(exc))
                return []
        client = self._client_or_none()
        if client is None:
            return []
        try:
            search = getattr(client, "search", None)
            if search is None:
                return []
            # Python SDK: client.search.memories(...). Some docs show client.search(...).
            if callable(search) and not hasattr(search, "memories"):
                payload = search(
                    q=query,
                    container_tag=container_tag,
                    search_mode=search_mode,
                    limit=5,
                )
            else:
                payload = search.memories(
                    q=query,
                    container_tag=container_tag,
                    search_mode=search_mode,
                    limit=5,
                )
            return _search_results(payload)
        except Exception as exc:  # noqa: BLE001 — never crash the debate loop
            self._deactivate(str(exc))
            return []

    def _add(self, content: str, *, container_tag: str, metadata: dict[str, Any] | None = None) -> bool:
        if self.is_active and self.base_url in {
            "http://localhost:6767",
            "http://127.0.0.1:6767",
        }:
            payload: dict[str, Any] = {
                "content": content,
                "container_tag": container_tag,
            }
            if metadata:
                payload["metadata"] = metadata
            custom_id = str((metadata or {}).get("custom_id") or "").strip()
            if custom_id:
                payload["custom_id"] = custom_id
            try:
                import requests

                response = requests.post(
                    f"{self.base_url}/v3/memories",
                    headers=(
                        {"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key
                        else {}
                    ),
                    json=payload,
                    timeout=REQUEST_TIMEOUT_S,
                )
                response.raise_for_status()
                return True
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("Supermemory add failed (%s)", exc)
                return False
        client = self._client_or_none()
        if client is None:
            return False
        payload: dict[str, Any] = {"content": content, "container_tag": container_tag}
        if metadata:
            payload["metadata"] = metadata
        custom_id = str((metadata or {}).get("custom_id") or "").strip()
        if custom_id:
            payload["custom_id"] = custom_id
        try:
            client.add(**payload)
            return True
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Supermemory add failed (%s)", exc)
            return False

    def _load_ledger(self) -> dict[str, Any]:
        path = self.ledger_path
        try:
            from modules.durable_store import hydrate_state_file

            path = hydrate_state_file(path)
        except Exception:
            pass
        if not Path(path).is_file():
            return {"sessions": []}
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("history ledger unreadable (%s); starting empty", exc)
            return {"sessions": []}
        if isinstance(raw, list):
            return {"sessions": [row for row in raw if isinstance(row, dict)]}
        sessions = raw.get("sessions") if isinstance(raw, dict) else []
        if not isinstance(sessions, list):
            sessions = []
        return {"sessions": [row for row in sessions if isinstance(row, dict)]}

    def _save_ledger(self, sessions: list[dict[str, Any]]) -> None:
        payload = {"sessions": sessions}
        if self._persist_durable:
            try:
                from modules.durable_store import write_state_json

                dest = write_state_json("aiwake", LEDGER_FILENAME, payload)
                self.ledger_path = Path(dest)
                return
            except Exception:
                pass
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.ledger_path)

    def ledger_sessions(self) -> list[dict[str, Any]]:
        return list(self._load_ledger().get("sessions") or [])

    def _local_duplicate(self, topic_query: str) -> bool:
        query = (topic_query or "").strip()
        if not query:
            return False
        for row in self.ledger_sessions():
            for field in ("topic", "hook", "quote"):
                candidate = str(row.get(field) or "").strip()
                if candidate and lexical_similarity(query, candidate) >= self.similarity_threshold:
                    return True
        return False

    def check_topic_similarity(self, topic_query: str) -> bool:
        """True when *topic_query* is a near-duplicate of approved history."""
        query = (topic_query or "").strip()
        if not query:
            return False
        if self.is_active:
            hits = self._search(
                query,
                container_tag=self.history_container,
                search_mode="memories",
            )
            if not hits:
                # Local embeddings index document chunks even when memory
                # extraction (LLM dreaming) is unavailable.
                hits = self._search(
                    query,
                    container_tag=self.history_container,
                    search_mode="documents",
                )
            if hits:
                top = max(_hit_score(hit) for hit in hits)
                return top > self.similarity_threshold
            if self.is_active:
                return False
        return self._local_duplicate(query)

    def remember_approved_session(
        self,
        session_id: str,
        topic: str,
        hook: str,
        quote: str,
        *,
        remote: bool = True,
    ) -> bool:
        """Persist an approved debate footprint. Always updates the local ledger."""
        sid = (session_id or "").strip()
        topic_text = (topic or "").strip()
        hook_text = (hook or "").strip()
        quote_text = (quote or "").strip()
        if not sid and not topic_text:
            return False

        sessions = self.ledger_sessions()
        existing = next((row for row in sessions if str(row.get("session_id") or "") == sid), None)
        record = {
            "session_id": sid,
            "topic": topic_text,
            "hook": hook_text,
            "quote": quote_text,
        }
        if existing is None:
            sessions.append(record)
        else:
            existing.update(record)
        try:
            self._save_ledger(sessions)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("local history ledger write failed (%s)", exc)

        if remote and self.is_active:
            custom_id = re.sub(
                r"[^A-Za-z0-9_:-]+",
                "-",
                f"aiwake-{sid}" if sid else f"aiwake-topic-{topic_text[:40]}",
            )
            self._add(
                session_memory_text(
                    session_id=sid,
                    topic=topic_text,
                    hook=hook_text,
                    quote=quote_text,
                ),
                container_tag=self.history_container,
                metadata={
                    "session_id": sid,
                    "kind": "approved_session",
                    "topic": topic_text[:200],
                    "custom_id": custom_id[:100],
                },
            )
        return True

    def remember_system_rule(self, rule: str = GOLDEN_RULE) -> bool:
        text = (rule or GOLDEN_RULE).strip()
        if not text:
            return False
        if not self.is_active:
            return False
        return self._add(
            text,
            container_tag=self.rules_container,
            metadata={"kind": "system_rule", "rule_id": "caption_contract", "custom_id": "aiwake-system-rule"},
        )


def get_bridge() -> SupermemoryBridge:
    """Process-wide bridge. Isolated during pytest unless SUPERMEMORY_IN_TESTS=1."""
    global _BRIDGE
    if _BRIDGE is None:
        in_pytest = bool(os.getenv("PYTEST_CURRENT_TEST"))
        force_offline = _env_flag("SUPERMEMORY_DISABLED") or (
            in_pytest and not _env_flag("SUPERMEMORY_IN_TESTS")
        )
        _BRIDGE = SupermemoryBridge(force_offline=force_offline)
    return _BRIDGE


def reset_bridge() -> None:
    """Drop the cached singleton (tests)."""
    global _BRIDGE, _START_ATTEMPTED
    _BRIDGE = None
    _START_ATTEMPTED = False


def check_topic_similarity(topic_query: str) -> bool:
    try:
        return get_bridge().check_topic_similarity(topic_query)
    except Exception as exc:  # noqa: BLE001 — pipeline must never crash
        _LOG.warning("topic similarity check failed (%s); treating as unique", exc)
        return False


def remember_approved_session(session_id: str, topic: str, hook: str, quote: str) -> bool:
    try:
        return get_bridge().remember_approved_session(session_id, topic, hook, quote)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("remember_approved_session failed (%s)", exc)
        return False


__all__ = [
    "GOLDEN_RULE",
    "HISTORY_CONTAINER",
    "LEDGER_FILENAME",
    "OFFLINE_WARNING",
    "RULES_CONTAINER",
    "SIMILARITY_THRESHOLD",
    "SupermemoryBridge",
    "check_topic_similarity",
    "ensure_supermemory_running",
    "get_bridge",
    "lexical_similarity",
    "remember_approved_session",
    "reset_bridge",
    "session_memory_text",
]
