from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from app_pages.shared import (clusters, csv_bytes, reasons, require_raw, rules, sota_index, sticky, store,
                               subset_key, vectors)
from sota import data, text, viz
from sota.store import StoreError, make_meta, owner_id

st.header("Explorar el estado del arte")
raw = require_raw()
meta = st.session_state.get("sota_meta")
if meta:
    st.info(f"📚 **{meta['name']}** — {meta.get('description', '')}  \n"
            f"Publicado por {meta.get('author') or 'anónimo'} el {meta.get('created', '?')}.")
kept = (reasons(raw) == "").to_numpy()
df = raw[kept].reset_index(drop=True)
if len(df) < 10:
    st.warning("Muy pocos artículos tras la limpieza.")
    st.stop()

# Re-vectorize the clean corpus so the vocabulary reflects only relevant docs.
key = subset_key(kept, "explore")
v = vectors(df, key)


@st.cache_resource(max_entries=4, ttl=3 * 3600, show_spinner="Preparando el mapa…")
def knn(_Z, key):
    return text.knn_graph(_Z)


@st.cache_data(max_entries=12, ttl=3 * 3600, show_spinner="Calculando el mapa…")
def project(_Z, key, separation, _init):
    return text.project_2d(_Z, separation, knn(_Z, key), _init)


HAS_SAVED_MAP = {"x", "y"} <= set(df.columns) and df[["x", "y"]].notna().all().all()
ss = st.session_state
ss.setdefault("map_sep", (meta or {}).get("map_sep", text.SEPARATION))


def map_xy():
    """Saved map of a published SOTA until the user asks to adjust it; then computed (cached)."""
    if HAS_SAVED_MAP and not ss.get("map_custom"):
        return df[["x", "y"]].to_numpy()
    init = ss.get("map_prev", df[["x", "y"]].to_numpy() if HAS_SAVED_MAP else None)
    xy, _ = project(v.Z, key, round(ss.map_sep, 3), init)
    ss.map_prev = xy  # next adjustment starts from this layout
    return xy


def adjust_map(more_space: bool) -> None:
    """Bisection on the separation between the user's 'too tight' and 'too spread' answers."""
    lo, hi = ss.get("map_lo", 0.0), ss.get("map_hi", 1.0)
    if more_space:
        lo = ss.map_sep
    else:
        hi = ss.map_sep
    ss.map_lo, ss.map_hi, ss.map_sep = lo, hi, (lo + hi) / 2
    ss.map_custom = True


with st.sidebar:
    st.subheader("Tópicos y filtros")
    k = sticky(st.slider, "Número de tópicos", "k_topics", meta["k"] if meta else text.suggest_k(len(df)),
               min_value=2, max_value=40)
    years = df["publication_year"].dropna().astype(int)
    yr = st.slider("Años", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))
    df["tipo"] = data.doc_type(df)
    types = st.multiselect("Tipo", sorted(df["tipo"].unique()), placeholder="Todos")

labels = clusters(v, key, k)
names = text.topic_names(v, labels)
df["tópico"] = pd.Series(labels).map(names).values

f = df["publication_year"].between(*yr).fillna(False).to_numpy()
if types:
    f &= df["tipo"].isin(types).to_numpy()
d = df[f]
st.caption(f"Mostrando **{f.sum():,}** de {len(df):,} artículos limpios.")

SECTIONS = ["Resumen", "Tópicos", "Mapa", "Búsqueda", "Autores y fuentes", "Términos", "Datos"]
# Only the selected section runs (st.tabs would compute all of them, including UMAP).
sec = sticky(st.segmented_control, "Sección", "explore_view", SECTIONS[0], options=SECTIONS) or SECTIONS[0]

if sec == "Resumen":
    c = st.columns(4)
    c[0].metric("Artículos", f"{len(d):,}")
    c[1].metric("Años", f"{yr[0]}–{yr[1]}")
    c[2].metric("Fuentes distintas", f"{d['publication_title'].nunique():,}")
    c[3].metric("Citas (mediana)", f"{d['article_citation_count'].median():.0f}")
    st.plotly_chart(viz.per_year(d, d["tipo"], title="Publicaciones por año y tipo"), width="stretch")
    st.subheader("Más citados")
    st.dataframe(d.nlargest(20, "article_citation_count")[
                     ["document_title", "publication_year", "publication_title", "article_citation_count", "html_url"]],
                 hide_index=True, column_config={"html_url": st.column_config.LinkColumn("link", display_text="abrir")})

elif sec == "Tópicos":
    groups = st.slider("Grupos de tópicos (jerarquía)", 1, max(1, k // 2), max(1, k // 4))
    h = text.topic_hierarchy(v.Z, labels, groups)
    gnames = {g: f"Grupo {g}" for g in set(h.values())}
    tree = pd.DataFrame({"grupo": pd.Series(labels[f]).map(h).map(gnames), "tópico": d["tópico"].values})
    st.plotly_chart(viz.treemap(tree.value_counts().rename("docs").reset_index(), ["grupo", "tópico"],
                                "Jerarquía de tópicos (tamaño = nº de artículos)"), width="stretch")
    share = st.toggle("Ver como % del año")
    st.plotly_chart(viz.per_year(d, d["tópico"], share, "Evolución de tópicos"), width="stretch")
    top_venues = d["publication_title"].value_counts().head(12).index
    tv = pd.crosstab(d["tópico"], d["publication_title"]).reindex(columns=top_venues)
    st.plotly_chart(viz.heatmap(tv, "Tópicos × fuentes principales", zlabel="docs"), width="stretch")

    st.subheader("Artículo representativo de cada tópico")
    rep = []
    for t in np.unique(labels):
        idx = np.flatnonzero(labels == t)
        cen = v.Z[idx].mean(axis=0)
        i = idx[np.argmax(v.Z[idx] @ cen)]
        rep.append({"tópico": names[int(t)], "docs": len(idx), "representativo": df.at[i, "document_title"],
                    "año": df.at[i, "publication_year"], "link": df.at[i, "html_url"]})
    st.dataframe(pd.DataFrame(rep), hide_index=True,
                 column_config={"link": st.column_config.LinkColumn(display_text="abrir")})

elif sec == "Mapa":
    st.caption("Cada punto es un artículo; cercanía = similitud de contenido. Tamaño = citas. "
               "Selecciona puntos (lazo/caja) para listarlos.")
    xy = map_xy()
    color_by = st.radio("Color", ["tópico", "tipo", "década"], horizontal=True)
    col = (df["publication_year"] // 10 * 10).astype(str) if color_by == "década" else df[color_by]
    ev = st.plotly_chart(viz.topic_map(d, xy[f], col[f], "Mapa de artículos"),
                         width="stretch", on_select="rerun", selection_mode=("lasso", "box"))
    if not ss.get("map_ok"):
        st.write("**¿Cómo se ven los puntos?** El mapa se ajusta con tu respuesta.")
        c1, c2, c3 = st.columns(3)
        c1.button("🔍 Muy juntos", width="stretch", on_click=adjust_map, args=(True,))
        c2.button("👍 Están bien", width="stretch", type="primary", on_click=lambda: ss.update(map_ok=True))
        c3.button("↔️ Muy separados", width="stretch", on_click=adjust_map, args=(False,))
        if ss.get("map_hi", 1.0) - ss.get("map_lo", 0.0) < 0.07:
            st.caption("Ya está casi en el límite de ajuste en esa dirección.")
    else:
        st.button("Reajustar separación de los puntos", on_click=lambda: ss.update(map_ok=False))
    pts = ev.selection.points if ev else []
    if pts:
        ids = [p["customdata"][0] for p in pts]
        st.dataframe(d[d["id"].isin(ids)][["document_title", "publication_year", "tópico", "html_url"]],
                     hide_index=True, column_config={"html_url": st.column_config.LinkColumn(display_text="abrir")})

elif sec == "Búsqueda":
    st.caption("Escribe una o varias consultas (una por línea). Se buscan los artículos más similares "
               "semánticamente y se muestra cómo se reparten entre los tópicos.")
    qs = [q.strip() for q in st.text_area("Consultas", "modular multilevel converter\nmodel predictive control").splitlines() if q.strip()]
    n_nb = st.slider("Vecinos por consulta", 10, 200, 50)
    if qs:
        sim = text.similar(v, qs)
        sim[:, ~f] = -np.inf
        prof = {}
        for q, s in zip(qs, sim):
            top = np.argsort(-s)[:n_nb]
            prof[q] = pd.Series(labels[top]).map(names).value_counts(normalize=True) * 100
            with st.expander(f"🔎 {q}"):
                st.dataframe(df.iloc[top][["document_title", "publication_year", "tópico", "html_url"]]
                             .assign(similitud=s[top].round(3)), hide_index=True,
                             column_config={"html_url": st.column_config.LinkColumn(display_text="abrir")})
        st.plotly_chart(viz.heatmap(pd.DataFrame(prof).T.fillna(0), "Perfil temático de cada consulta (% de vecinos)",
                                    zlabel="%"), width="stretch")

elif sec == "Autores y fuentes":
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.top_bar(data.split_list(d["authors"]), title="Autores con más artículos"), width="stretch")
    c2.plotly_chart(viz.top_bar(d["publication_title"], title="Fuentes principales"), width="stretch")
    aff = data.split_list(d["author_affiliations"])
    c1.plotly_chart(viz.top_bar(aff, title="Afiliaciones"), width="stretch")
    c2.plotly_chart(viz.top_bar(aff.map(data.country), title="Países (aprox., según afiliación)"), width="stretch")
    st.plotly_chart(viz.citations(d, d["tópico"], "Citas por año de publicación"), width="stretch")

elif sec == "Términos":
    recent = st.slider("Ventana 'reciente' (años)", 2, 10, 5)
    em = text.emerging_terms(v.subset(f), d["publication_year"], recent)
    st.plotly_chart(viz.emerging(em, f"Términos emergentes (últimos {recent} años vs. antes)"), width="stretch")
    w = np.asarray(v.X[f].sum(axis=0)).ravel()
    st.plotly_chart(viz.hbar(pd.Series(w, v.terms).nlargest(25), "Términos con más peso TF-IDF", "Peso TF-IDF"),
                    width="stretch")

elif sec == "Datos":
    st.dataframe(d.drop(columns=["id"]), hide_index=True, height=500)
    st.download_button("⬇️ Descargar (.csv con tópicos)", csv_bytes(d), "sota_topics.csv", "text/csv")

if st.session_state.get("api_key"):
    st.divider()
    with st.expander("📤 Publicar este SOTA para que cualquiera pueda verlo"):
        if not store().writable:
            st.caption("La publicación no está configurada en este servidor (falta el token de GitHub en *secrets*).")
        else:
            m0 = meta or {}
            name = st.text_input("Nombre", m0.get("name", ""), placeholder="p. ej. Electromovilidad marítima")
            desc = st.text_area("Descripción", m0.get("description", ""))
            author = st.text_input("Autor(es)", m0.get("author", ""))
            st.caption(f"Se guardan los {len(df):,} artículos limpios, el mapa, la regla de limpieza y el número de "
                       "tópicos, en Parquet comprimido. Solo tu API key puede volver a publicar con este nombre.")
            if st.button("Publicar", type="primary", disabled=not name.strip()):
                xy = map_xy()
                out = df.assign(x=xy[:, 0], y=xy[:, 1])
                m = make_meta(out, name, desc, author, owner_id(st.session_state.api_key), k, asdict(rules()))
                m["map_sep"] = round(ss.map_sep, 3)
                try:
                    with st.spinner("Publicando…"):
                        store().save(out, m)
                except StoreError as e:
                    st.error(str(e))
                else:
                    sota_index.clear()
                    st.session_state.sota_meta = m
                    st.success(f"Publicado: **{m['name']}**. Ya aparece en *SOTA publicados*.")
