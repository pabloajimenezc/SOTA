"""Session state and cached computations shared by the pages."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

from sota import cleaning, data, text
from sota.cleaning import Rules
from sota.ieee import IEEEClient
from sota.store import GitHubStore

DEFAULT_REPO = "pabloajimenezc/SOTA"


def client() -> IEEEClient:
    # Built per request from session memory; never cached (cache is shared across users).
    return IEEEClient(st.session_state.api_key)


def rules() -> Rules:
    return st.session_state.setdefault("rules", Rules())


@st.cache_resource
def store() -> GitHubStore:
    """Published SOTAs. Configure [github] repo/branch/token in .streamlit/secrets.toml to enable publishing."""
    try:
        gh = dict(st.secrets.get("github", {}))
    except Exception:  # no secrets file (local run): read-only from the public repo
        gh = {}
    return GitHubStore(gh.get("repo", DEFAULT_REPO), gh.get("branch", "sotas"), gh.get("token"))


@st.cache_data(ttl=300, show_spinner=False)
def sota_index() -> list[dict]:
    return store().index()


@st.cache_data(ttl=3 * 3600, max_entries=4, show_spinner="Cargando SOTA…")
def sota_load(slug: str, version: str) -> pd.DataFrame:
    return data.add_links(store().load(slug))


STICKY = ["topic_path", "topic_path_k", "k_topics", "k_clean", "explore_view", "clean_view", "map_sep", "map_lo", "map_hi", "map_ok", "map_custom"]


def sticky(widget, label: str, key: str, default, **kw):
    """Widget whose value survives page switches (Streamlit drops widget state when the page changes)."""
    st.session_state.setdefault(key, default)
    if f"_w_{key}" not in st.session_state:
        st.session_state[f"_w_{key}"] = st.session_state[key]
    st.session_state[key] = widget(label, key=f"_w_{key}", **kw)
    return st.session_state[key]


def set_raw(df: pd.DataFrame) -> None:
    st.session_state.pop("sota_meta", None)  # own data from now on
    for k in STICKY:  # settings belong to the previous dataset
        st.session_state.pop(k, None)
        st.session_state.pop(f"_w_{k}", None)
    st.session_state.raw = df
    st.session_state.raw_key = hashlib.md5(pd.util.hash_pandas_object(df["id"], index=False).values).hexdigest()


def require_raw() -> pd.DataFrame:
    raw = st.session_state.get("raw")
    if raw is None or raw.empty:
        st.info("Primero carga artículos en **1 · Buscar** o abre un SOTA publicado.")
        if "api_key" in st.session_state:
            st.page_link("app_pages/search.py", label="Ir a Buscar", icon="🔎")
        st.page_link("app_pages/gallery.py", label="Ver SOTA publicados", icon="📚")
        st.stop()
    return raw


def _mask_key(mask: np.ndarray) -> str:
    return hashlib.md5(np.packbits(mask).tobytes()).hexdigest()


@st.cache_resource(max_entries=4, ttl=3 * 3600, show_spinner="Vectorizando textos…")
def _vectors(_df: pd.DataFrame, key: str, synonyms: str) -> text.Vectors:
    return text.vectorize(_df, json.loads(synonyms))


def vectors(df: pd.DataFrame, key: str) -> text.Vectors:
    return _vectors(df, key, json.dumps(rules().synonyms, sort_keys=True))


@st.cache_data(max_entries=4, ttl=3 * 3600, show_spinner=False)
def _haystack(_df: pd.DataFrame, key: str) -> pd.Series:
    return cleaning.haystack(_df)


@st.cache_data(max_entries=16, ttl=3 * 3600, show_spinner=False)
def _reasons(_raw: pd.DataFrame, key: str, rules_json: str) -> pd.Series:
    return cleaning.apply(_raw, Rules.from_json(rules_json), _haystack(_raw, key))


def reasons(raw: pd.DataFrame) -> pd.Series:
    """Removal reason per row of raw ('' = kept)."""
    return _reasons(raw, st.session_state.raw_key, rules().to_json())


@st.cache_data(max_entries=8, ttl=3 * 3600, show_spinner="Agrupando…")
def _clusters(_Z: np.ndarray, key: str, k: int) -> np.ndarray:
    return text.cluster(_Z, k)


def clusters(v: text.Vectors, key: str, k: int) -> np.ndarray:
    """`key` must identify the rows of v (see subset_key) so the cache is not mixed up."""
    return _clusters(v.Z, key, k)


def subset_key(mask: np.ndarray, tag: str = "") -> str:
    """Cache key for a subset of raw rows (+ synonyms, which change the vectors)."""
    return f"{tag}:{st.session_state.raw_key}:{_mask_key(mask)}:{json.dumps(rules().synonyms, sort_keys=True)}"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
