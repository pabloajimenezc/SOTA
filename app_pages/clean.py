from pathlib import Path

import numpy as np
import streamlit as st

from app_pages.shared import clusters, csv_bytes, reasons, require_raw, rules, sticky, subset_key, vectors
from sota import cleaning, text
from sota.cleaning import Rules

RULES_DIR = Path(__file__).resolve().parents[1] / "rules"

st.header("2 · Limpieza")
raw = require_raw()
R = rules()
hist = st.session_state.setdefault("history", [])  # [(attr, values)] for undo


def act(changes: list[tuple[str, list]], msg: str) -> None:
    done = [(a, [v for v in vals if v not in getattr(R, a)]) for a, vals in changes]
    for a, vals in done:
        R.add(a, vals)
    hist.append(done)
    st.session_state.flash = msg
    st.rerun()


if msg := st.session_state.pop("flash", None):
    st.toast(msg)

# ---------------- Rule editor (sidebar) ----------------
with st.sidebar:
    st.subheader("🧾 Regla de limpieza")
    syntax = "Palabra completa, `prefijo*` o frase. Sin distinguir mayúsculas."
    ex = st.multiselect("Excluir si contiene", R.exclude_terms, R.exclude_terms, accept_new_options=True,
                        help=syntax, placeholder="Escribe un término y Enter")
    req = st.multiselect("Mantener solo si contiene alguno", R.require_terms, R.require_terms,
                         accept_new_options=True, help=syntax + " Vacío = no se exige nada.",
                         placeholder="Opcional")
    min_year = st.number_input("Año mínimo (0 = sin filtro)", 0, 2100, R.min_year or 0)
    syn = st.text_input("Sinónimos (afecta clusters y tópicos)",
                        ", ".join(f"{a}={b}" for a, b in R.synonyms.items()), placeholder="vessel=ship, ocean=sea")
    R.exclude_terms, R.require_terms, R.min_year = list(ex), list(req), int(min_year) or None
    R.synonyms = dict(p.split("=", 1) for p in (s.strip().lower() for s in syn.split(",")) if "=" in p)

    st.download_button("⬇️ Descargar regla (.json)", R.to_json(), f"{R.name}.json", "application/json",
                       width="stretch")
    up = st.file_uploader("Cargar regla (.json)", type="json")
    if up and up.file_id != st.session_state.get("rules_file"):
        st.session_state.rules_file = up.file_id
        st.session_state.rules = Rules.from_json(up.getvalue())
        st.rerun()
    presets = sorted(RULES_DIR.glob("*.json"))
    if presets:
        p = st.selectbox("Ejemplos", presets, format_func=lambda p: p.stem.replace("_", " "))
        if st.button("Cargar ejemplo", width="stretch"):
            st.session_state.rules = Rules.from_json(p.read_text(encoding="utf-8"))
            st.rerun()
    if hist and st.button("↩️ Deshacer última acción", width="stretch"):
        for a, vals in hist.pop():
            setattr(R, a, [x for x in getattr(R, a) if x not in vals])
        st.rerun()

# ---------------- Current state ----------------
reason = reasons(raw)
kept = (reason == "").to_numpy()
dk = raw[kept].reset_index(drop=True)
c1, c2, c3 = st.columns(3)
c1.metric("Descargados", f"{len(raw):,}")
c2.metric("Se mantienen", f"{kept.sum():,}")
c3.metric("Eliminados", f"{(~kept).sum():,}")
if len(dk) < 10:
    st.warning("Quedan muy pocos artículos para agrupar. Revisa la regla.")
    st.stop()

v = vectors(raw, st.session_state.raw_key).subset(kept)
key = subset_key(kept)

with st.expander("🔬 Probar un término antes de añadirlo"):
    t = st.text_input("Término", placeholder="p. ej. plasma*  o  blood vessel")
    if t:
        hit = cleaning.matches(dk, [t]).notna()
        st.write(f"**{hit.sum()}** artículos lo contienen:")
        st.dataframe(dk.loc[hit, ["document_title", "publication_year", "publication_title"]].head(50),
                     hide_index=True)
        if hit.any() and st.button("Añadir a términos excluidos"):
            act([("exclude_terms", [t.strip().lower()])], f"Añadido '{t}' ({hit.sum()} eliminados)")

VIEWS = ["Clusters", "Revisión uno a uno", "Eliminados"]
view = sticky(st.segmented_control, "Vista", "clean_view", VIEWS[0], options=VIEWS) or VIEWS[0]


def term_picker(terms: list[str], wkey: str, scope=None) -> list[str]:
    picked = st.pills("Términos (selecciona para añadir a la regla)", terms, selection_mode="multi", key=wkey)
    if picked:
        hit = cleaning.matches(dk, picked).notna().to_numpy()
        extra = f" ({(hit & scope).sum()} de este grupo, {(hit & ~scope).sum()} de otros)" if scope is not None else ""
        st.caption(f"Excluir {', '.join(picked)} eliminaría **{hit.sum()}** artículos{extra}.")
    return picked or []


# ---------------- Clusters ----------------
if view == "Clusters":
    st.caption("Clusters ordenados del más **distinto** al resto (arriba suelen estar los temas ajenos, "
               "p. ej. medicina por *vessel*). Selecciona uno para inspeccionarlo.")
    k = sticky(st.slider, "Número de clusters", "k_clean", text.suggest_k(len(dk)), min_value=2, max_value=40)
    labels = clusters(v, key, k)
    summ = text.cluster_summary(v, labels, dk)
    sel = st.dataframe(summ, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"summ{k}",
                       column_config={"distinto": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    if sel.selection.rows:
        c = int(summ.loc[sel.selection.rows[0], "cluster"])
        m = labels == c
        st.subheader(f"Cluster {c} · {m.sum()} artículos")
        picked = term_picker(text.distinctive_terms(v, labels, 15)[c], f"pc{key}{c}", m)
        b1, b2, b3 = st.columns(3)
        if b1.button("➕ Añadir términos a la regla", disabled=not picked):
            act([("exclude_terms", picked)], f"Añadidos {len(picked)} términos")
        if b2.button(f"🗑️ Eliminar cluster completo ({m.sum()})"):
            act([("exclude_ids", dk.loc[m, "id"].tolist())], f"Eliminados {m.sum()} artículos")

        def review_cluster(ids=dk.loc[m, "id"].tolist()):
            st.session_state.review_ids = ids
            st.session_state.clean_view = st.session_state._w_clean_view = "Revisión uno a uno"

        b3.button("🔍 Revisar uno a uno", on_click=review_cluster)
        st.dataframe(dk.loc[m, ["document_title", "publication_year", "publication_title", "ieee_terms"]],
                     hide_index=True, height=350)

# ---------------- One by one ----------------
elif view == "Revisión uno a uno":
    ids = st.session_state.get("review_ids")
    scope_all = st.radio("Revisar", ["Todos (más atípicos primero)", "Cluster seleccionado"],
                         index=1 if ids else 0, horizontal=True, disabled=not ids) != "Cluster seleccionado"
    reviewed = set(R.keep_ids) | set(R.exclude_ids) | st.session_state.setdefault("skipped", set())
    order = np.argsort(-text.atypicality(v.Z))
    in_scope = np.ones(len(dk), bool) if scope_all else dk["id"].isin(ids).to_numpy()
    queue = [i for i in order if in_scope[i] and dk.at[i, "id"] not in reviewed]
    st.progress(1 - len(queue) / max(in_scope.sum(), 1), f"Quedan {len(queue)} de {in_scope.sum()} por revisar")
    if not queue:
        st.success("No quedan artículos por revisar en este grupo.")
        if st.button("Volver a mostrar los saltados"):
            st.session_state.skipped = set()
            st.rerun()
    else:
        i = queue[0]
        row = dk.iloc[i]
        url = row["html_url"] if str(row["html_url"]).startswith("http") else row["pdf_link"]
        st.markdown(f"#### [{row['document_title']}]({url})")
        st.caption(f"{row['publication_year']} · {row['publication_title']} · {row['document_identifier']} · "
                   f"{row['article_citation_count']} citas · atipicidad {text.atypicality(v.Z)[i]:.2f}")
        st.write(row["abstract"])
        st.caption(f"**IEEE terms:** {row['ieee_terms'] or '—'}  \n**Keywords:** {row['author_keywords'] or '—'}")
        picked = term_picker(text.doc_terms(v, i), f"pd{row['id']}")
        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Relevante", type="primary", width="stretch"):
            act([("keep_ids", [row["id"]])], "Marcado como relevante")
        label = "🗑️ Eliminar" + (" + añadir términos" if picked else "")
        if b2.button(label, width="stretch"):
            act([("exclude_ids", [row["id"]]), ("exclude_terms", picked)], "Eliminado")
        if b3.button("⏭️ Saltar", width="stretch"):
            st.session_state.skipped.add(row["id"])
            st.rerun()
        st.caption("*Relevante* protege el artículo de futuros términos de exclusión. Todo queda guardado en la regla.")

# ---------------- Removed ----------------
elif view == "Eliminados":
    rem = raw[~kept].assign(motivo=reason[~kept].values).reset_index(drop=True)
    if rem.empty:
        st.info("Aún no se ha eliminado nada.")
    else:
        st.bar_chart(rem["motivo"].value_counts().head(25), horizontal=True)
        sel = st.dataframe(rem[["motivo", "document_title", "publication_year", "publication_title"]],
                           hide_index=True, on_select="rerun", selection_mode="multi-row", height=400)
        rows = sel.selection.rows
        if st.button(f"↩️ Restaurar seleccionados ({len(rows)})", disabled=not rows):
            ids = rem.loc[rows, "id"].tolist()
            R.exclude_ids = [x for x in R.exclude_ids if x not in ids]
            act([("keep_ids", ids)], f"Restaurados {len(ids)}")

st.divider()
c1, c2 = st.columns(2)
c1.download_button("⬇️ Descargar datos limpios (.csv)", csv_bytes(dk), "sota_clean.csv", "text/csv")
c2.page_link("app_pages/explore.py", label="Siguiente: explorar →", icon="📊")
