"""Minimal client for the IEEE Xplore Metadata Search API.

Docs: https://developer.ieee.org/docs/read/Metadata_API_details

Notes verified against the live API:
- Without sort params results come by relevance (field ``rank``).
- ``sort_field`` only supports article_number / article_title / publication_title;
  publication_year and citation counts are silently ignored. "Newest"/"oldest"
  are therefore built by splitting the year range, "most cited" by downloading
  everything and sorting locally.
- ``content_type`` accepts a single value per call.

The API key is never logged, printed or included in error messages.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass
from typing import Callable

import requests

ENDPOINT = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
PAGE = 200  # API maximum per call

CONTENT_TYPES = ["Journals", "Conferences", "Early Access", "Magazines", "Books", "Standards", "Courses"]

SEARCH_FIELDS = {
    "Todos los metadatos": "querytext",
    "Abstract": "abstract",
    "Título": "article_title",
    "Index terms": "index_terms",
}

# label -> (sort key, API params)
SORTS = {
    "Relevancia": ("relevance", {}),
    "Más recientes": ("newest", {}),
    "Más antiguos": ("oldest", {}),
    "Más citados": ("cited", {}),
    "Título de publicación (A-Z)": ("api", {"sort_field": "publication_title", "sort_order": "asc"}),
    "Título de publicación (Z-A)": ("api", {"sort_field": "publication_title", "sort_order": "desc"}),
    "Título de artículo (A-Z)": ("api", {"sort_field": "article_title", "sort_order": "asc"}),
}


class IEEEError(RuntimeError):
    pass


@dataclass(frozen=True)
class Query:
    text: str
    field: str = "querytext"
    start_year: int | None = None
    end_year: int | None = None
    open_access: bool = False

    def params(self) -> dict:
        p = {self.field: self.text}
        if self.start_year:
            p["start_year"] = self.start_year
        if self.end_year:
            p["end_year"] = self.end_year
        if self.open_access:
            p["open_access"] = "True"
        return p


def _explain(r: requests.Response) -> str:
    text = re.sub(r"<[^>]+>", " ", r.text or "").strip()[:200]
    if r.status_code in (401, 403):
        if "Over" in text:
            return f"Límite de uso de la API alcanzado: {text}"
        return "API key inválida o inactiva (IEEE respondió 403)."
    return f"IEEE Xplore respondió {r.status_code}: {text}"


def allocate(n: int, counts: dict) -> dict:
    """Split n proportionally to counts (largest remainder)."""
    total = sum(counts.values())
    if total <= n:
        return dict(counts)
    raw = {k: n * c / total for k, c in counts.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


class IEEEClient:
    def __init__(self, api_key: str, session: requests.Session | None = None, pause: float = 0.12):
        self._key = api_key
        self._s = session or requests.Session()
        self.pause = pause  # API allows ~10 calls/s
        self.calls = 0
        self._tick: Callable[[int], None] = lambda k: None

    def __repr__(self) -> str:
        return "IEEEClient(api_key=***)"

    def _get(self, params: dict) -> dict:
        try:
            r = self._s.get(ENDPOINT, params={"apikey": self._key, "format": "json", **params}, timeout=60)
        except requests.RequestException as e:
            # `from None` drops the original exception, whose message contains the URL with the key.
            raise IEEEError(f"No se pudo conectar con IEEE Xplore ({type(e).__name__}).") from None
        self.calls += 1
        time.sleep(self.pause)
        if r.status_code != 200:
            raise IEEEError(_explain(r))
        try:
            return r.json()
        except ValueError:
            raise IEEEError("Respuesta inválida de IEEE Xplore (¿API key correcta?).") from None

    def _params(self, q: Query, ctype: str | None, **extra) -> dict:
        p = q.params() | extra
        if ctype:
            p["content_type"] = ctype
        return p

    def validate(self) -> None:
        """Raises IEEEError if the key does not work."""
        self._get({"querytext": "power", "max_records": 1})

    def count(self, q: Query, ctype: str | None = None) -> int:
        return int(self._get(self._params(q, ctype, max_records=1)).get("total_records", 0))

    def counts(self, q: Query, types: list[str] | None = None) -> dict:
        """Total per content type (key None = no filter)."""
        return {t: self.count(q, t) for t in (types or [None])}

    @staticmethod
    def estimate_calls(n: int, total: int, sort: str) -> int:
        k = SORTS[sort][0]
        m = total if k == "cited" else min(n, total)
        return -(-m // PAGE) + (8 if k in ("newest", "oldest") else 0)

    def search(self, q: Query, n: int, sort: str = "Relevancia", counts: dict | None = None,
               progress: Callable[[int], None] | None = None) -> list[dict]:
        """Download up to n records. `counts` is the output of `counts()` (reused to avoid calls)."""
        counts = counts or self.counts(q)
        kind, sort_params = SORTS[sort]
        done = [0]

        def tick(k):
            done[0] += k
            if progress:
                progress(done[0])

        self._tick = tick
        out = []
        for ctype, m in allocate(n, counts).items():
            if not m:
                continue
            p = self._params(q, ctype, **sort_params)
            if kind in ("newest", "oldest"):
                lo, hi = q.start_year or 1900, q.end_year or dt.date.today().year + 1
                out += self._by_year(p, lo, hi, m, newest=kind == "newest")
            elif kind == "cited":
                recs = self._paginate(p, counts[ctype])
                out += sorted(recs, key=lambda a: a.get("citing_paper_count") or 0, reverse=True)[:m]
            else:
                out += self._paginate(p, m)

        # Merge content types into a single ordering.
        if kind == "newest":
            out.sort(key=lambda a: -int(a.get("publication_year") or 0))
        elif kind == "oldest":
            out.sort(key=lambda a: int(a.get("publication_year") or 9999))
        elif kind == "cited":
            out.sort(key=lambda a: a.get("citing_paper_count") or 0, reverse=True)
        elif kind == "relevance":
            out.sort(key=lambda a: a.get("rank") or 0)
        return out

    def _paginate(self, p: dict, m: int, first: dict | None = None) -> list[dict]:
        recs = list(first.get("articles", [])) if first else []
        total = first.get("total_records", 0) if first else None
        if recs:
            self._tick(len(recs))
        while len(recs) < m and (total is None or len(recs) < total):
            page = self._get(p | {"max_records": PAGE, "start_record": len(recs) + 1})
            total = page.get("total_records", 0)
            batch = page.get("articles", [])
            if not batch:
                break
            recs += batch
            self._tick(len(batch))
        return recs[:m]

    def _by_year(self, p: dict, lo: int, hi: int, m: int, newest: bool) -> list[dict]:
        """Top-m records by year, splitting [lo, hi] until each chunk fits in m."""
        first = self._get(p | {"start_year": lo, "end_year": hi, "max_records": PAGE, "start_record": 1})
        total = int(first.get("total_records", 0))
        if total == 0:
            return []
        if total <= m or lo == hi:
            return self._paginate(p | {"start_year": lo, "end_year": hi}, m, first)
        mid = (lo + hi) // 2
        halves = [(mid + 1, hi), (lo, mid)] if newest else [(lo, mid), (mid + 1, hi)]
        out = []
        for a, b in halves:
            out += self._by_year(p, a, b, m - len(out), newest)
            if len(out) >= m:
                break
        return out
