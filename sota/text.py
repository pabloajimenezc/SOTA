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
    k = max(1, min(k, len(Z)))
    return KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(Z)


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
            "cluster": int(c), "docs": int(m.sum()), "distinto": round(float(1 - cen @ glob), 3),
            "términos": ", ".join(terms[int(c)]),
            "ejemplo": df.loc[m, "document_title"].iloc[0],
        })
    return pd.DataFrame(rows).sort_values("distinto", ascending=False).reset_index(drop=True)


def topic_names(v: Vectors, labels: np.ndarray, n: int = 3) -> dict[int, str]:
    return {c: f"T{c}: " + ", ".join(t[:n]) for c, t in distinctive_terms(v, labels, n).items()}


def similar(v: Vectors, queries: list[str]) -> np.ndarray:
    """Cosine similarity (queries x docs)."""
    return v.embed(queries) @ v.Z.T


def project_2d(Z: np.ndarray) -> tuple[np.ndarray, str]:
    """UMAP if available (better clusters), else the first 2 LSA components."""
    try:
        import umap
        n = len(Z)
        xy = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.1, metric="cosine",
                       random_state=0, low_memory=True).fit_transform(Z)
        return xy, "UMAP"
    except Exception:
        return Z[:, :2], "LSA"


def topic_hierarchy(Z: np.ndarray, labels: np.ndarray, groups: int) -> dict[int, int]:
    """Group topics (Ward linkage over topic centroids). Returns topic -> group."""
    from scipy.cluster.hierarchy import fcluster, linkage
    topics = np.unique(labels)
    if len(topics) <= 2:
        return {int(t): 1 for t in topics}
    cen = normalize(np.vstack([Z[labels == t].mean(axis=0) for t in topics]))
    g = fcluster(linkage(cen, "ward"), t=min(groups, len(topics)), criterion="maxclust")
    return dict(zip(map(int, topics), map(int, g)))


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
