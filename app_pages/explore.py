import hashlib
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from app_pages.shared import (clusters, csv_bytes, published, reasons, require_raw, rules, sota_index, sticky,
                               store, subset_key, vectors)
from sota import data, geo, text, viz
from sota.store import StoreError, make_meta, owner_id

raw = require_raw()
meta = st.session_state.get("sota_meta")
if meta:
    st.title(meta["name"])
    st.caption(f"{meta.get('description', '')}  \nPublicado por **{meta.get('author') or 'anónimo'}** · "
               f"{meta.get('created', '?')}")
else:
    st.title("Tu estado del arte")
kept = (reasons(raw) == "").to_numpy()
df = raw[kept].reset_index(drop=True)
if len(df) < 10:
    st.warning("Muy pocos artículos tras la limpieza.")
    st.stop()

# Re-vectorize the clean corpus so the vocabulary reflects only relevant docs.
key = subset_key(kept, "explore")
v = vectors(df, key)


@st.cache_data(max_entries=512, ttl=3 * 3600, show_spinner="Subdividiendo…")
def children(_v, key, node_key, _idx):
    return text.split(_v, _idx)


@st.cache_data(max_entries=8, ttl=3 * 3600, show_spinner="Ubicando instituciones…")
def affiliations(_df, key):
    return geo.affiliations(_df)


@st.cache_resource(max_entries=4, ttl=3 * 3600, show_spinner="Preparando el mapa…")
def knn(_Z, key):
    return text.knn_graph(_Z)


@st.cache_data(max_entries=12, ttl=3 * 3600, show_spinner="Calculando el mapa…")
def project(_Z, key, separation, _init):
    return text.project_2d(_Z, separation, knn(_Z, key), _init)


XY_SAVED = df[["x", "y"]].to_numpy(dtype=float) if {"x", "y"} <= set(df.columns) else np.full((len(df), 2), np.nan)
HAS_SAVED_MAP = not np.isnan(XY_SAVED).any()
ss = st.session_state
ss.setdefault("map_sep", (meta or {}).get("map_sep", text.SEPARATION))


def base_map() -> np.ndarray:
    """Reference layout: the saved one, or computed deterministically (anchored on saved positions if any)."""
    if HAS_SAVED_MAP:
        return XY_SAVED
    xy, _ = project(v.Z, key, text.SEPARATION, text.anchored_init(v.Z, XY_SAVED))
    return xy


def map_xy():
    """Base layout until the user adjusts spacing; adjusted maps always start from the base so they stay aligned."""
    base = base_map()
    if not ss.get("map_custom"):
        return base
    xy, _ = project(v.Z, key, round(ss.map_sep, 3), base)
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
    st.subheader("Ajustes")
    k = sticky(st.number_input, "Cantidad de temas", "k_topics", meta["k"] if meta else text.suggest_k(len(df)),
               min_value=2, max_value=60, step=1,
               help="En cuántos grandes temas se agrupan los artículos. Más temas = grupos más específicos.")
    years = df["publication_year"].dropna().astype(int)
    yr = st.slider("Años", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))
    df["tipo"] = data.doc_type(df)
    types = st.multiselect("Tipo de publicación", sorted(df["tipo"].unique()), placeholder="Todos")
    aff = affiliations(df, key)
    doc_iso = geo.article_countries(aff, len(df))
    per_country = pd.Series([c for cs in doc_iso for c in cs]).value_counts()
    isos = sorted(per_country.index, key=lambda c: geo.es_key(geo.country_name(c)))  # Spanish alphabetical order
    iso = st.selectbox("País destacado", isos, index=isos.index("CL") if "CL" in isos else None,
                       format_func=lambda c: f"{geo.country_name(c)} ({per_country[c]:,})", placeholder="Ninguno",
                       help="Sus artículos se marcan con su bandera en el mapa. Según la afiliación de los autores.")
    country = geo.country_name(iso) if iso else ""
    df["destacado"] = [iso in cs for cs in doc_iso]
    only_cl = st.toggle(f"Solo artículos de {country}" if iso else "Solo artículos del país destacado",
                        disabled=not iso)

pub = published(df)
if pub and k == pub.get("k") and "topic" in df and df["topic"].notna().all():
    labels = df["topic"].to_numpy(dtype=int)  # saved at publish time
else:
    labels = clusters(v, key, k)
names = text.topic_names(v, labels)
df["tema"] = pd.Series(labels).map(names).values

f = df["publication_year"].between(*yr).fillna(False).to_numpy()
if types:
    f &= df["tipo"].isin(types).to_numpy()
if only_cl:
    f &= df["destacado"].to_numpy()
d = df[f]
if d.empty:
    st.warning("Ningún artículo cumple los filtros. Amplía los años o quita filtros.")
    st.stop()

c = st.columns(5)
c[0].metric("Artículos", f"{len(d):,}")
c[1].metric("Desde", f"{yr[0]}")
c[2].metric("Temas", k)
c[3].metric("Fuentes", f"{d['publication_title'].nunique():,}")
c[4].metric(country or "País", f"{int(d['destacado'].sum()):,}" if iso else "—")

LINK = {"html_url": st.column_config.LinkColumn("enlace", display_text="abrir ↗"),
        "destacado": st.column_config.CheckboxColumn(country or "país"),
        "document_title": st.column_config.TextColumn("título", width="large"),
        "publication_year": st.column_config.NumberColumn("año", format="%d"),
        "publication_title": "revista / congreso", "article_citation_count": "citas"}
SECTIONS = ["📊 Resumen", "🧭 Temas", "🗺️ Mapa", "🌍 Geografía", "🔎 Buscar", "👥 Autores", "📈 Tendencias",
            "📄 Datos"]
# Only the selected section runs (st.tabs would compute all of them, including the map).
sec = sticky(st.segmented_control, "Sección", "explore_view", SECTIONS[0], options=SECTIONS,
             label_visibility="collapsed") or SECTIONS[0]

if sec == SECTIONS[0]:
    st.plotly_chart(viz.per_year(d, d["tipo"], title="Publicaciones por año"), width="stretch")
    st.subheader("Artículos más citados")
    st.dataframe(d.nlargest(20, "article_citation_count")[
                     ["document_title", "publication_year", "publication_title", "article_citation_count", "destacado",
                      "html_url"]], hide_index=True, column_config=LINK)

elif sec == SECTIONS[1]:
    st.caption("Cada cuadro es un tema; su tamaño es la cantidad de artículos. **Haz clic en un cuadro** para "
               "ver sus subtemas, y así hasta llegar a un artículo.")
    if ss.get("topic_path_k") != k:  # path refers to the current topic count
        ss.topic_path, ss.topic_path_k = [], k
    path = ss.topic_path

    # Walk down the tree: root = the k topics, below = automatic subdivisions (cached per node).
    groups = [g for g in (np.flatnonzero((labels == t) & f) for t in range(k))]
    codes = [str(t + 1) for t in range(k)]
    tags = [names[t].split(": ", 1)[1] for t in range(k)]
    trail = []
    for depth, pick in enumerate(path):
        if pick >= len(groups) or len(groups[pick]) == 0:
            del path[depth:]
            break
        node, code = groups[pick], codes[pick]
        trail.append(code)
        groups = children(v, key, hashlib.md5(node.tobytes()).hexdigest(), node)
        codes = [f"{code}.{j + 1}" for j in range(len(groups))]
        tags = [df.at[g[0], "document_title"][:80] if len(g) == 1 else ", ".join(t)
                for g, t in zip(groups, text.group_terms(v, groups) if groups else [])]
    keep = [i for i, g in enumerate(groups) if len(g)]
    groups, codes, tags = [groups[i] for i in keep], [codes[i] for i in keep], [tags[i] for i in keep]

    shown = list(enumerate(trail))[-4:]  # breadcrumb: last levels only
    crumbs = st.columns(5)
    if crumbs[0].button("🏠 Todos los temas", disabled=not path, width="stretch"):
        path.clear()
        st.rerun()
    for col, (i, code) in zip(crumbs[1:], shown):
        if col.button(f"› {code}", disabled=i == len(trail) - 1, key=f"crumb_{code}", width="stretch"):
            del path[i + 1:]
            st.rerun()

    if not groups:  # a single article
        row = df.iloc[node[0]]
        with st.container(border=True):
            st.markdown(f"#### {'🏳 ' if row['destacado'] else ''}[{row['document_title']}]({row['html_url']})")
            st.caption(f"{row['publication_year']} · {row['publication_title']} · {row['article_citation_count']} citas"
                       f"  \n{row['authors']}")
            st.write(row["abstract"])
    else:
        n_node = sum(map(len, groups))
        title = f"Tema {trail[-1]} · {n_node:,} artículos" if trail else f"{k} temas · {n_node:,} artículos"
        ev = st.plotly_chart(viz.treemap_flat([len(g) for g in groups], codes, tags, title), width="stretch",
                             on_select="rerun", selection_mode="points", key=f"tree_{'.'.join(trail)}_{k}")
        clicked = [p["point_index"] for p in (ev.selection.points if ev else [])]
        if clicked:
            path.append(keep[clicked[0]])
            st.rerun()

        grp = pd.Series(index=np.concatenate(groups), dtype="object")
        for c, t, g in zip(codes, tags, groups):
            grp[g] = f"{c}: {t[:40]}"
        dn = df.loc[grp.index].assign(grupo=grp.values)
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        c1.subheader("¿Cómo evolucionan?")
        share = c2.toggle("Ver como % de cada año")
        st.plotly_chart(viz.per_year(dn, dn["grupo"], share), width="stretch")
        top_venues = dn["publication_title"].value_counts().head(12).index
        tv = pd.crosstab(dn["grupo"], dn["publication_title"]).reindex(columns=top_venues)
        st.subheader("¿Dónde se publican?")
        st.plotly_chart(viz.heatmap(tv, zlabel="artículos"), width="stretch")

        st.subheader("Un artículo representativo de cada uno")
        rep = []
        for c, t, g in zip(codes, tags, groups):
            i = g[np.argmax(v.Z[g] @ v.Z[g].mean(axis=0))]
            rep.append({"tema": c, "palabras clave": t, "artículos": len(g),
                        "document_title": df.at[i, "document_title"], "publication_year": df.at[i, "publication_year"],
                        "html_url": df.at[i, "html_url"]})
        st.dataframe(pd.DataFrame(rep), hide_index=True, column_config=LINK)

elif sec == SECTIONS[2]:
    st.caption("Cada punto es un artículo: **los parecidos quedan cerca**. Tamaño = citas. "
               f"{'Las banderas marcan artículos con autores de ' + country + '. ' if iso else ''}"
               "Encierra puntos con el lazo para listarlos.")
    xy = map_xy()
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    color_by = c1.segmented_control("Colorear por", ["Tema", "Tipo", "Década"], default="Tema")
    mark_cl = c2.toggle("Mostrar banderas", value=True, disabled=not iso)
    col = {"Tema": df["tema"], "Tipo": df["tipo"],
           "Década": (df["publication_year"] // 10 * 10).astype(str)}[color_by or "Tema"]
    ev = st.plotly_chart(viz.topic_map(d, xy[f], col[f], highlight=d["destacado"] if mark_cl and iso else None, flag=iso),
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
        st.dataframe(d[d["id"].isin(ids)][["document_title", "publication_year", "tema", "destacado", "html_url"]],
                     hide_index=True, column_config=LINK)

elif sec == SECTIONS[3]:
    st.caption("Cada punto es una institución (universidad, centro o empresa) con artículos en este SOTA; su "
               "tamaño es la cantidad de artículos. **Haz clic en un punto** (o encierra varios) para ver sus "
               "publicaciones. Ubicación aproximada: ciudad de la afiliación.")
    aff_d = aff[f[aff["doc"].to_numpy()]]
    inst = geo.institutions(aff_d)
    inst["destacado"] = inst["iso"] == iso
    ev = st.plotly_chart(viz.world(inst, country), width="stretch", on_select="rerun",
                         selection_mode=("points", "box", "lasso"), key=f"world_{iso}_{yr}_{len(d)}")
    rows = sorted({int(p["customdata"][0]) for p in (ev.selection.points if ev else []) if p.get("customdata")})
    if rows:
        picked = inst.iloc[rows]
        keys = set(zip(picked["institution"].map(geo.norm), picked["iso"]))
        docs = sorted({doc for doc, i, c in zip(aff_d["doc"], aff_d["institution"].map(geo.norm), aff_d["iso"])
                       if (i, c) in keys})
        one = len(picked) == 1
        st.subheader(picked["institution"].iloc[0] if one else f"{len(picked)} instituciones")
        place = f"{picked['city'].iloc[0] or ''}, {picked['country'].iloc[0]} · " if one else ""
        st.caption(f"{place}{len(docs)} artículos en este SOTA")
        st.dataframe(df.iloc[docs].sort_values("publication_year", ascending=False)[
                         ["document_title", "publication_year", "tema", "article_citation_count", "html_url"]],
                     hide_index=True, column_config=LINK)
    else:
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.hbar(geo.country_counts(aff_d).head(12), "Países con más artículos", "Artículos"),
                        width="stretch")
        c2.plotly_chart(viz.hbar(inst.set_index("institution")["docs"].head(12), "Instituciones con más artículos",
                                 "Artículos"), width="stretch")

elif sec == SECTIONS[4]:
    st.caption("Escribe uno o varios temas (uno por línea), en inglés. Encontramos los artículos más parecidos "
               "y te mostramos en qué temas del SOTA caen.")
    c1, c2 = st.columns([3, 1])
    qs = [q.strip() for q in c1.text_area("¿Qué buscas?", "modular multilevel converter\nmodel predictive control",
                                          height=110).splitlines() if q.strip()]
    n_nb = c2.number_input("Resultados por búsqueda", 10, 200, 30, step=10)
    if qs:
        sim = text.similar(v, qs)
        sim[:, ~f] = -np.inf
        prof = {}
        for q, s in zip(qs, sim):
            top = np.argsort(-s)[:n_nb]
            prof[q] = pd.Series(labels[top]).map(names).value_counts(normalize=True) * 100
            with st.expander(f"🔎 {q}", expanded=len(qs) == 1):
                st.dataframe(df.iloc[top][["document_title", "publication_year", "tema", "destacado", "html_url"]]
                             .assign(parecido=(s[top] * 100).round().astype(int)), hide_index=True,
                             column_config=LINK | {"parecido": st.column_config.ProgressColumn(
                                 "parecido", min_value=0, max_value=100, format="%d%%")})
        st.subheader("¿En qué temas cae cada búsqueda?")
        st.plotly_chart(viz.heatmap(pd.DataFrame(prof).T.fillna(0), zlabel="% de resultados"), width="stretch")

elif sec == SECTIONS[5]:
    aff_d = aff[f[aff["doc"].to_numpy()]]
    inst_all = geo.institutions(aff_d)
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.top_bar(data.split_list(d["authors"]), title="Autores con más artículos"), width="stretch")
    c2.plotly_chart(viz.top_bar(d["publication_title"], title="Revistas y congresos principales"), width="stretch")
    c1.plotly_chart(viz.hbar(inst_all.set_index("institution")["docs"].head(15), "Instituciones", "Artículos"),
                    width="stretch")
    c2.plotly_chart(viz.hbar(geo.country_counts(aff_d).head(15), "Países", "Artículos"), width="stretch")
    if iso:
        inst_c = inst_all[inst_all["iso"] == iso]
        if len(inst_c):
            st.plotly_chart(viz.hbar(inst_c.set_index("institution")["docs"].head(15),
                                     f"Instituciones de {country}", "Artículos"), width="stretch")
    st.plotly_chart(viz.citations(d, d["tema"], "Citas según año de publicación"), width="stretch")

elif sec == SECTIONS[6]:
    st.caption("Palabras que aparecen cada vez más en los artículos recientes: pistas de hacia dónde va el área.")
    recent = st.slider("¿Cuántos años cuentan como recientes?", 2, 10, 5)
    em = text.emerging_terms(v.subset(f), d["publication_year"], recent)
    st.plotly_chart(viz.emerging(em, f"Palabras en alza (últimos {recent} años vs. antes)"), width="stretch")
    w = np.asarray(v.X[f].sum(axis=0)).ravel()
    st.plotly_chart(viz.hbar(pd.Series(w, v.terms).nlargest(25), "Palabras más características del área",
                             "Importancia"), width="stretch")

elif sec == SECTIONS[7]:
    st.dataframe(d.drop(columns=["id", "x", "y", "topic"], errors="ignore"), hide_index=True, height=520,
                 column_config=LINK)
    st.download_button("⬇️ Descargar tabla (.csv)", csv_bytes(d), "sota.csv", "text/csv", type="primary")

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
            st.caption(f"Se guardan los {len(df):,} artículos limpios ya procesados (mapa, temas y textos), para "
                       "que abran en segundos. Solo tu API key puede volver a publicar con este nombre.")
            if st.button("Publicar", type="primary", disabled=not name.strip()):
                xy = map_xy()
                out = df.assign(x=xy[:, 0], y=xy[:, 1], topic=labels)
                m = make_meta(out, name, desc, author, owner_id(st.session_state.api_key), k, asdict(rules()))
                m["map_sep"] = round(ss.map_sep, 3)
                m["chile"] = int(data.chile(df).sum())
                try:
                    with st.spinner("Publicando…"):
                        store().save(out, m, text.pack_vectors(v))
                except StoreError as e:
                    st.error(str(e))
                else:
                    sota_index.clear()
                    st.session_state.sota_meta = m
                    st.success(f"Publicado: **{m['name']}**. Ya aparece en *SOTA publicados*.")
