"""Published SOTAs: one compressed Parquet per SOTA + index.json, stored on a GitHub branch.

A separate branch (default "sotas") keeps publications from redeploying the app.
Reads work without a token on a public repo; writes need a token with Contents read/write.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import io
import json
import re
import unicodedata

import pandas as pd
import requests

from sota.data import COLUMNS

SAVE_COLUMNS = COLUMNS + ["id", "html_url", "query", "x", "y"]
MAX_BYTES = 25_000_000


class StoreError(RuntimeError):
    pass


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:60] or "sota"


def owner_id(api_key: str) -> str:
    """Anonymous publisher id: only the same IEEE key can overwrite its SOTAs."""
    return hashlib.sha256(f"sota-explorer:{api_key}".encode()).hexdigest()[:16]


def pack(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df[[c for c in SAVE_COLUMNS if c in df]].to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def make_meta(df: pd.DataFrame, name: str, description: str, author: str, owner: str, k: int, rules: dict) -> dict:
    years = df["publication_year"].dropna().astype(int)
    return {
        "slug": slugify(name), "name": name.strip(), "description": description.strip(), "author": author.strip(),
        "owner": owner, "created": dt.date.today().isoformat(), "docs": len(df),
        "years": [int(years.min()), int(years.max())] if len(years) else None,
        "queries": sorted({q for q in df.get("query", pd.Series(dtype=str)).dropna() if q}),
        "k": int(k), "rules": rules,
    }


class GitHubStore:
    def __init__(self, repo: str, branch: str = "sotas", token: str | None = None,
                 session: requests.Session | None = None):
        self.repo, self.branch, self._token = repo, branch, token
        self._s = session or requests.Session()

    def __repr__(self) -> str:
        return f"GitHubStore({self.repo}@{self.branch}, token=***)"

    @property
    def writable(self) -> bool:
        return bool(self._token)

    def _headers(self, raw: bool = False) -> dict:
        h = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _req(self, method: str, url: str, **kw) -> requests.Response:
        try:
            return self._s.request(method, url, timeout=60, **kw)
        except requests.RequestException as e:
            raise StoreError(f"No se pudo conectar con GitHub ({type(e).__name__}).") from None

    def _read(self, path: str) -> bytes | None:
        if self._token:  # API: no CDN cache, works on private repos
            r = self._req("GET", f"https://api.github.com/repos/{self.repo}/contents/{path}",
                          params={"ref": self.branch}, headers=self._headers(raw=True))
        else:
            r = self._req("GET", f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/{path}")
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise StoreError(f"GitHub respondió {r.status_code} al leer {path}.")
        return r.content

    def _write(self, path: str, content: bytes, message: str) -> None:
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"
        r = self._req("GET", url, params={"ref": self.branch}, headers=self._headers())
        sha = r.json().get("sha") if r.status_code == 200 else None
        body = {"message": message, "content": base64.b64encode(content).decode(), "branch": self.branch}
        if sha:
            body["sha"] = sha
        r = self._req("PUT", url, json=body, headers=self._headers())
        if r.status_code not in (200, 201):
            raise StoreError(f"GitHub respondió {r.status_code} al guardar {path}.")

    def index(self) -> list[dict]:
        raw = self._read("index.json")
        return json.loads(raw) if raw else []

    def load(self, slug: str) -> pd.DataFrame:
        raw = self._read(f"sotas/{slug}.parquet")
        if raw is None:
            raise StoreError("Ese SOTA ya no existe.")
        return pd.read_parquet(io.BytesIO(raw))

    def save(self, df: pd.DataFrame, meta: dict) -> None:
        if not self.writable:
            raise StoreError("Publicación deshabilitada: falta configurar el token de GitHub.")
        idx = self.index()
        old = next((m for m in idx if m["slug"] == meta["slug"]), None)
        if old and old.get("owner") != meta["owner"]:
            raise StoreError("Ya existe un SOTA con ese nombre publicado por otra persona. Usa otro nombre.")
        blob = pack(df)
        if len(blob) > MAX_BYTES:
            raise StoreError(f"El SOTA ocupa {len(blob) / 1e6:.0f} MB (máximo {MAX_BYTES / 1e6:.0f} MB).")
        meta = meta | {"bytes": len(blob)}
        self._write(f"sotas/{meta['slug']}.parquet", blob, f"Publish SOTA: {meta['name']}")
        idx = [m for m in idx if m["slug"] != meta["slug"]] + [meta]
        self._write("index.json", json.dumps(idx, indent=1, ensure_ascii=False).encode(), f"Index: {meta['name']}")
