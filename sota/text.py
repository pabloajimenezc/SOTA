"""Text preprocessing, vectorization (TF-IDF + LSA), clustering and search."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

TEXT_FIELDS = ["document_title", "abstract", "ieee_terms", "author_keywords", "mesh_terms"]
STOP = ENGLISH_STOP_WORDS | {
    "proposed", "proposal", "method", "achieve", "algorithm", "also", "base", "based", "paper",
    "article", "study", "result", "results", "approach", "using", "used", "use", "show", "shown",
    "present", "presented", "work", "ieee", "copyright", "new", "novel",
}
TOKEN = re.compile(r"[a-z][a-z0-9]+")


@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        lem = WordNetLemmatizer()
        lem.lemmatize("ships")
        return lru_cache(maxsize=200_000)(lem.lemmatize)
    except Exception:  # offline / no nltk: crude plural stripping
        return lambda w: w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w


def clean(text: str, synonyms: dict | None = None) -> str:
    lem, syn = _lemmatizer(), synonyms or {}
    out = []
    for w in TOKEN.findall(str(text).lower()):
        if w in STOP:
            continue
        w = lem(w)
        w = syn.get(w, w)
        if w not in STOP:
            out.append(w)
    return " ".join(out)


def corpus(df: pd.DataFrame, synonyms: dict | None = None) -> list[str]:
    raw = df[TEXT_FIELDS].fillna("").astype(str).agg(" ".join, axis=1)
    return [clean(t, synonyms) for t in raw]


@dataclass
class Vectors:
    X: sparse.csr_matrix      # TF-IDF (docs x terms)
    Z: np.ndarray             # LSA embedding, L2-normalized (docs x k)
    terms: np.ndarray
    tfidf: TfidfVectorizer
    svd: TruncatedSVD
    synonyms: dict

    def embed(self, texts: list[str]) -> np.ndarray:
        X = self.tfidf.transform([clean(t, self.synonyms) for t in texts])
        return normalize(self.svd.transform(X))

    def subset(self, mask) -> "Vectors":
        idx = np.flatnonzero(mask)
        return Vectors(self.X[idx], self.Z[idx], self.terms, self.tfidf, self.svd, self.synonyms)


def vectorize(df: pd.DataFrame, synonyms: dict | None = None, dims: int = 100) -> Vectors:
    texts = corpus(df, synonyms)
    n = len(texts)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=2 if n < 1000 else 5,
                            max_features=20000, sublinear_tf=True)
    X = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=max(2, min(dims, X.shape[1] - 1, n - 1)), random_state=0)
    Z = normalize(svd.fit_transform(X))
    return Vectors(X.tocsr(), Z, tfidf.get_feature_names_out(), tfidf, svd, synonyms or {})


def suggest_k(n: int) -> int:
    return int(np.clip(round(np.sqrt(n / 4)), 2, 30))


def cluster(Z: np.ndarray, k: int) -> np.ndarray:
    """Deterministic KMeans; label 0 = largest cluster, so numbering is stable."""
    k = max(1, min(k, len(Z)))
    lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(Z)
    order = np.argsort(-np.bincount(lab, minlength=k), kind="stable")
    return np.argsort(order)[lab]


def distinctive_terms(v: Vectors, labels: np.ndarray, n: int = 8) -> dict[int, list[str]]:
    """Terms over-represented in each cluster vs the whole corpus."""
    glob = np.asarray(v.X.mean(axis=0)).ravel()
    out = {}
    for c in np.unique(labels):
        mean = np.asarray(v.X[labels == c].mean(axis=0)).ravel()
        out[int(c)] = v.terms[np.argsort(mean - glob)[::-1][:n]].tolist()
    return out


def doc_terms(v: Vectors, i: int, n: int = 12) -> list[str]:
    row = v.X[i]
    return v.terms[row.indices[np.argsort(row.data)[::-1][:n]]].tolist()


def atypicality(Z: np.ndarray) -> np.ndarray:
    """Cosine distance of each doc to the corpus centroid (0 = typical, 1 = unrelated)."""
    c = normalize(Z.mean(axis=0, keepdims=True))
    return 1 - (Z @ c.T).ravel()


def cluster_summary(v: Vectors, labels: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster, most distinct (far from the corpus centroid) first."""
    terms = distinctive_terms(v, labels)
    glob = normalize(v.Z.mean(axis=0, keepdims=True)).ravel()
    rows = []
    for c in np.unique(labels):
        m = labels == c
        cen = normalize(v.Z[m].mean(axis=0, keepdims=True)).ravel()
        rows.append({
            "cluster": int(c) + 1, "docs": int(m.sum()), "distinto": round(float(1 - cen @ glob), 3),
            "términos": ", ".join(terms[int(c)]),
            "ejemplo": df.loc[m, "document_title"].iloc[0],
        })
    return pd.DataFrame(rows).sort_values("distinto", ascending=False).reset_index(drop=True)


def topic_names(v: Vectors, labels: np.ndarray, n: int = 3) -> dict[int, str]:
    return {c: f"T{c + 1}: " + ", ".join(t[:n]) for c, t in distinctive_terms(v, labels, n).items()}


def split(v: Vectors, idx: np.ndarray) -> list[np.ndarray]:
    """Subdivide a set of docs (row indices of v) into subtopics, largest first. Ends at single docs."""
    n = len(idx)
    if n <= 1:
        return []
    lab = cluster(v.Z[idx], int(np.clip(round(np.sqrt(n / 2)), 2, 8)))
    groups = [idx[lab == c] for c in np.unique(lab)]
    if len(groups) == 1:  # identical vectors: one doc per group
        groups = [idx[i:i + 1] for i in range(n)]
    return sorted(groups, key=len, reverse=True)


def group_terms(v: Vectors, groups: list[np.ndarray], n: int = 4) -> list[list[str]]:
    """Terms that distinguish each group from its siblings."""
    parent = np.asarray(v.X[np.concatenate(groups)].mean(axis=0)).ravel()
    return [v.terms[np.argsort(np.asarray(v.X[g].mean(axis=0)).ravel() - parent)[::-1][:n]].tolist()
            for g in groups]


def similar(v: Vectors, queries: list[str]) -> np.ndarray:
    """Cosine similarity (queries x docs)."""
    return v.embed(queries) @ v.Z.T


UMAP_NEIGHBORS = 30
SEPARATION = 0.5  # 0 = tight clumps, 1 = evenly spread


def _neighbors(Z: np.ndarray) -> int:
    return max(2, min(UMAP_NEIGHBORS, len(Z) - 1))


def knn_graph(Z: np.ndarray):
    """kNN graph shared by every UMAP run on the same docs (the slow part of UMAP)."""
    from umap.umap_ import nearest_neighbors
    # Seeded: NN-descent is random otherwise, which would change the map on every session.
    idx, dist, _ = nearest_neighbors(Z, _neighbors(Z), "cosine", {}, False, np.random.RandomState(0))
    return idx, dist


def anchored_init(Z: np.ndarray, xy: np.ndarray) -> np.ndarray | None:
    """Start layout from known positions (NaN rows = new docs, placed on their most similar known doc).

    Keeps the map recognizable when a SOTA gains or loses articles.
    """
    known = ~np.isnan(xy).any(axis=1)
    if known.sum() < max(10, 0.5 * len(Z)):
        return None
    init = xy.copy()
    unk = np.flatnonzero(~known)
    ref = np.flatnonzero(known)
    for s in range(0, len(unk), 2000):
        part = unk[s:s + 2000]
        init[part] = xy[ref[np.argmax(Z[part] @ Z[ref].T, axis=1)]]
    span = np.nanstd(xy, axis=0) * 0.01
    return init + np.random.RandomState(0).normal(size=init.shape) * span


def project_2d(Z: np.ndarray, separation: float = SEPARATION, knn=None, init: np.ndarray | None = None
               ) -> tuple[np.ndarray, str]:
    """UMAP if available (better clusters), else the first 2 LSA components.

    `separation` hides UMAP's knobs: it drives min_dist and spread together.
    `init` (a previous layout) keeps the map stable between adjustments.
    """
    try:
        import umap
        kw = {"precomputed_knn": (*knn, None)} if knn is not None else {}
        xy = umap.UMAP(n_neighbors=_neighbors(Z), min_dist=0.99 * separation, spread=1.0 + 1.5 * separation,
                       metric="cosine", random_state=0, low_memory=True,
                       init=init if init is not None and len(init) == len(Z) else "spectral", **kw).fit_transform(Z)
        return xy, "UMAP"
    except ImportError:
        return Z[:, :2], "LSA"


def emerging_terms(v: Vectors, years: pd.Series, recent: int = 5, n: int = 20, min_docs: int = 5) -> pd.DataFrame:
    """Terms whose document share grew the most in the last `recent` years."""
    y = years.to_numpy(dtype=float, na_value=np.nan)
    cut = np.nanmax(y) - recent + 1
    new, old = y >= cut, y < cut
    if new.sum() == 0 or old.sum() == 0:
        return pd.DataFrame(columns=["término", "antes", "reciente", "crecimiento"])
    B = (v.X > 0).astype(np.float32)
    f_new = np.asarray(B[new].mean(axis=0)).ravel()
    f_old = np.asarray(B[old].mean(axis=0)).ravel()
    count = np.asarray(B[new].sum(axis=0)).ravel()
    growth = (f_new + 1e-3) / (f_old + 1e-3)
    idx = [i for i in np.argsort(growth)[::-1] if count[i] >= min_docs][:n]
    return pd.DataFrame({"término": v.terms[idx], "antes": f_old[idx] * 100,
                         "reciente": f_new[idx] * 100, "crecimiento": growth[idx]})
