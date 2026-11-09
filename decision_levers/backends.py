"""Backends: the real TypeSafe Jev API (``JevBackend``) and an offline fake (``MockBackend``).

Both implement the ``Backend`` protocol: ``decide(request) -> {option: probability}``, a batch version and a
packed version. ``rep`` > 0 marks a deliberate identical resend (the A/A control): it gets its own cache entry.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import httpx

from .request import PackedRequest, Request, parse_answer

URL = "https://api.typesafe.ai/v1/systemone"
MODELS_URL = "https://api.typesafe.ai/v1/models"
PAPER_MODEL = "jev-1.13.0"                         # the version every result in the paper was measured on
KEY_ENVS = ("TYPESAFE_API_KEY", "JEV_API_KEY")     # the first is the one the research code read
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
Probs = dict[str, float]


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    model: str

    def decide(self, request: Request, rep: int = 0) -> Probs: ...
    def decide_many(self, requests: Sequence[Request], rep: int = 0) -> list[Probs]: ...
    def decide_packed(self, packed: PackedRequest, rep: int = 0) -> dict[str, Probs]: ...


def request_key(model: str, wire: dict, rep: int = 0) -> str:
    """SHA-256 of (model, wire body, repeat index). Key order is kept: option order is part of the request."""
    blob = json.dumps({"m": model, "w": wire, "rep": rep}, sort_keys=False, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def estimate_tokens(wire: dict) -> int:
    """Rough input-token count of a wire body (about 4 characters per token)."""
    return max(1, len(json.dumps(wire, ensure_ascii=False)) // 4)


class _BaseBackend:
    model: str = ""

    def __init__(self) -> None:
        self.stats = {"calls": 0, "cache_hits": 0, "input_tokens": 0}
        self._lock = threading.Lock()

    def _answers(self, wire: dict, rep: int) -> dict[str, Probs]:  # pragma: no cover - interface
        raise NotImplementedError

    def decide(self, request: Request, rep: int = 0) -> Probs:
        return self._answers(request.to_wire(self.model), rep)["q"]

    def decide_many(self, requests: Sequence[Request], rep: int = 0, workers: int = 8) -> list[Probs]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            return list(ex.map(lambda r: self.decide(r, rep), requests))

    def decide_packed(self, packed: PackedRequest, rep: int = 0) -> dict[str, Probs]:
        return self._answers(packed.to_wire(self.model), rep)


class JevBackend(_BaseBackend):
    """TypeSafe Jev over HTTPS with Bearer auth, retry with backoff, and an on-disk response cache.

    ``model`` must be given explicitly (e.g. ``PAPER_MODEL``); a response that reports another model id raises.
    The key is read from ``TYPESAFE_API_KEY`` (or ``JEV_API_KEY``) unless passed. Identical requests are
    answered from the cache and never paid twice.
    """

    def __init__(self, model: str, api_key: str | None = None, cache_dir: str | Path | None = None,
                 timeout: float = 90.0, retries: int = 6, client=None):
        super().__init__()
        if not model:
            raise ValueError("pin the model version explicitly, e.g. JevBackend(PAPER_MODEL)")
        self.model, self.retries = model, retries
        self._key = api_key or next((os.environ[k] for k in KEY_ENVS if os.environ.get(k)), "")
        if not self._key:
            raise BackendError(f"no API key: set one of {', '.join(KEY_ENVS)}")
        self.client = client or httpx.Client(timeout=timeout)
        d = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "decision-levers"
        d.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(d / "responses.sqlite"), timeout=60, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS r(key TEXT PRIMARY KEY, model TEXT, answers TEXT, tokens INT)")
        self._db.commit()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def _post(self, body: dict):
        delay, last = 1.0, None
        for attempt in range(self.retries + 1):
            try:
                r = self.client.post(URL, headers=self._headers(), json=body)
            except (httpx.HTTPError, OSError) as e:      # network blips: retry
                last = e
            else:
                if r.status_code not in RETRY_STATUS:
                    return r
                last = BackendError(f"HTTP {r.status_code}: {r.text[:200]}")
                ra = r.headers.get("retry-after", "")
                if ra.replace(".", "", 1).isdigit():
                    delay = max(delay, float(ra))
            if attempt < self.retries:
                time.sleep(min(delay, 30.0))
                delay *= 2
        raise BackendError(f"gave up after {self.retries + 1} attempts: {last}")

    def _answers(self, wire: dict, rep: int) -> dict[str, Probs]:
        key = request_key(self.model, wire, rep)
        with self._lock:
            row = self._db.execute("SELECT answers FROM r WHERE key=?", (key,)).fetchone()
        if row:
            self.stats["cache_hits"] += 1
            return json.loads(row[0])
        r = self._post(wire)
        if r.status_code != 200:
            raise BackendError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        if data.get("model", self.model) != self.model:
            raise BackendError(f"requested {self.model}, response says {data.get('model')}")
        ans = data.get("answers", {})
        missing = [q for q in wire["questions"] if q not in ans]
        if missing:
            raise BackendError(f"response lacks answers for {missing}")
        out = {q: parse_answer(ans[q]) for q in wire["questions"]}
        tokens = int(data.get("usage", {}).get("input_tokens", 0))
        with self._lock:
            self.stats["calls"] += 1
            self.stats["input_tokens"] += tokens
            self._db.execute("INSERT OR REPLACE INTO r VALUES(?,?,?,?)", (key, self.model, json.dumps(out), tokens))
            self._db.commit()
        return out

    def auth_check(self) -> bool:
        """Free: ``GET /v1/models`` confirms the key without spending anything."""
        return self.client.get(MODELS_URL, headers=self._headers()).status_code == 200


# ---------------------------------------------------------------------------------------------- offline fake

_STOP = set("the a an and or of to for in on at by with from as is are be was were it its this that my me i you your "
            "we our they them can could would should will not no yes do does did have has had what which who how "
            "please just about into than then there here".split())
_SHARE = re.compile(r"Approximate share of past cases by option: (.*)\.\s*$")


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "").lower())
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words if len(w) > 2 and w not in _STOP}


def _noise(seed: str, scale: float) -> float:
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16) / 16 ** 12
    return scale * (2 * h - 1)


class MockBackend(_BaseBackend):
    """Deterministic keyword-overlap fake, so every example runs offline for free.

    Each option scores the word overlap between the state and the option's name plus description, plus any
    ``knowledge`` words for that option NAME (a stand-in for what a pretrained model already knows; it vanishes
    when names are replaced by ids). A stated share sentence (``levers.state_shares``) adds ``log(share)``.
    Everything else in the instruction is ignored except for a small jitter seeded by the question and ``rep``,
    which is how content-free edits and resends behave on the real model. The fake mimics the DIRECTION of the
    paper's findings on purpose; its numbers are not evidence of anything.
    """

    def __init__(self, model: str = "mock-0", knowledge: Mapping[str, Iterable[str]] | None = None,
                 temperature: float = 0.35, jitter: float = 0.05):
        super().__init__()
        self.model, self.temperature, self.jitter = model, temperature, jitter
        self.knowledge = {o: _tokens(" ".join(ws)) for o, ws in (knowledge or {}).items()}

    def _choice(self, state: str, q: dict, seed: str) -> Probs:
        st = _tokens(state)
        shares: dict[str, float] = {}
        m = _SHARE.search(q.get("instructions") or "")
        if m:
            for part in m.group(1).split(", "):
                name, _, pct = part.rpartition(" ")
                shares[name] = float(pct.rstrip("%")) / 100 if pct.endswith("%") else 0.0
        logits = {}
        for o, desc in q["criteria"].items():
            ev = _tokens(o) | _tokens(desc or "")
            z = len(st & ev) / math.sqrt(len(ev) + 1.0) + 0.5 * len(st & self.knowledge.get(o, set()))
            if shares:
                z += 0.5 * math.log(shares.get(o, 0.0) + 0.02)
            # a fixed per-(state, option) quirk breaks ties the same way every time; the request-seeded jitter
            # is what an identical resend or a content-free edit can change
            logits[o] = (z / self.temperature + _noise(f"{state}/{o}", 0.5)
                         + _noise(f"{seed}/{o}", self.jitter))
        mx = max(logits.values())
        e = {o: math.exp(v - mx) for o, v in logits.items()}
        s = sum(e.values())
        return {o: round(v / s, 2) for o, v in e.items()}       # Jev also answers on a 0.01 grid

    def _answers(self, wire: dict, rep: int) -> dict[str, Probs]:
        with self._lock:
            self.stats["calls"] += 1
            self.stats["input_tokens"] += estimate_tokens(wire)
        out = {}
        for name, q in wire["questions"].items():
            # seeded per question, not per wire body: packing other questions alongside leaves an answer unchanged
            seed = request_key(self.model, {"s": wire["state"], "q": q}, rep)
            if q["type"] == "choice":
                out[name] = self._choice(wire["state"], q, seed)
            else:
                ov = len(_tokens(wire["state"]) & _tokens(q["instructions"]))
                p = round(1 / (1 + math.exp(-(1.5 * ov - 1.0) + _noise(seed, self.jitter))), 2)
                out[name] = {"true": p, "false": round(1 - p, 2)}
        return out
