import datetime as dt

import streamlit as st

from app_pages.shared import client, csv_bytes, set_raw
from sota import data, ieee

st.header("1 · Buscar artículos")
if msg := st.session_state.pop("flash", None):
    st.success(msg)
raw = st.session_state.get("raw")
year_now = dt.date.today().year

tab_api, tab_csv = st.tabs(["IEEE Xplore API", "Subir CSV"])

with tab_api:
    if not st.session_state.api_key:
        st.info("Ingresaste sin API key. Cierra sesión e ingresa con tu key para buscar, o usa **Subir CSV**.")
    else:
        with st.form("query"):
            q_text = st.text_input(
                "Tema de investigación",
                placeholder='("Abstract":"ship*" OR "Abstract":"vessel*") AND "Abstract":"propulsion"',
                help="Admite la sintaxis de IEEE Xplore: AND, OR, NOT, comillas, comodín *.")
            c1, c2, c3 = st.columns([1, 2, 2])
            field = c1.selectbox("Buscar en", list(ieee.SEARCH_FIELDS))
            types = c2.multiselect("Tipo de documento", ieee.CONTENT_TYPES, placeholder="Todos")
            years = c3.slider("Años", 1950, year_now + 1, (1950, year_now + 1))
            oa = st.checkbox("Solo Open Access")
            count_clicked = st.form_submit_button("Contar artículos", type="primary")

        if count_clicked and q_text.strip():
            q = ieee.Query(q_text.strip(), ieee.SEARCH_FIELDS[field],
                           years[0] if years[0] > 1950 else None,
                           years[1] if years[1] <= year_now else None, oa)
            try:
                with st.spinner("Consultando IEEE Xplore…"):
                    st.session_state.search = {"q": q, "counts": client().counts(q, types or None)}
            except ieee.IEEEError as e:
                st.error(str(e))

        s = st.session_state.get("search")
        if s:
            q, counts = s["q"], s["counts"]
            total = sum(counts.values())
            cols = st.columns(len(counts) + 1 if len(counts) > 1 else 1)
            cols[0].metric("Artículos encontrados", f"{total:,}")
            if len(counts) > 1:
                for col, (t, c) in zip(cols[1:], counts.items()):
                    col.metric(t, f"{c:,}")
            if total == 0:
                st.warning("Sin resultados. Prueba otra consulta o menos filtros.")
            else:
                c1, c2 = st.columns(2)
                n = c1.number_input("¿Cuántos descargar? (los N primeros)", 1, min(total, 50_000),
                                    min(total, 1000), step=100)
                sort = c2.selectbox("Ordenar por", list(ieee.SORTS))
                calls = ieee.IEEEClient.estimate_calls(n, total, sort)
                st.caption(f"≈ {calls} llamadas a la API (200 artículos por llamada; una key estándar "
                           "permite ~200 llamadas/día).")
                if sort == "Más citados":
                    st.warning(f"La API no ordena por citas: se descargan los {total:,} artículos y se "
                               "ordenan aquí.")
                if len(counts) > 1:
                    st.caption("Con varios tipos, N se reparte proporcionalmente al total de cada tipo.")
                merge = st.toggle("Combinar con los datos ya cargados", value=raw is not None,
                                  disabled=raw is None)
                if st.button("Descargar", type="primary"):
                    target = total if ieee.SORTS[sort][0] == "cited" else min(n, total)
                    bar = st.progress(0.0, "Descargando…")
                    cl = client()
                    try:
                        recs = cl.search(q, n, sort, counts,
                                         progress=lambda d: bar.progress(min(d / target, 1.0), f"{d:,} / {target:,}"))
                    except ieee.IEEEError as e:
                        st.error(str(e))
                    else:
                        df = data.from_api(recs, query=q.text)
                        set_raw(data.merge(raw, df) if merge else df)
                        st.session_state.flash = (f"{len(recs):,} descargados → {len(df):,} válidos (con abstract y sin "
                                   f"duplicados). Llamadas usadas: {cl.calls}.")
                        st.rerun()

with tab_csv:
    st.write("CSV exportados desde IEEE Xplore (*Export → CSV*) o descargados previamente desde esta app.")
    files = st.file_uploader("Archivos CSV", type="csv", accept_multiple_files=True)
    label = st.text_input("Etiqueta para estos archivos (opcional)", placeholder="p. ej. ShipPropulsion")
    merge_csv = st.toggle("Combinar con los datos ya cargados", value=raw is not None, disabled=raw is None,
                          key="merge_csv")
    if files and st.button("Cargar CSV", type="primary"):
        try:
            df = data.from_csv(files, label)
        except Exception as e:
            st.error(f"No se pudieron leer los archivos: {e}")
        else:
            set_raw(data.merge(raw, df) if merge_csv else df)
            st.rerun()

raw = st.session_state.get("raw")
if raw is not None:
    st.divider()
    st.subheader(f"Datos cargados: {len(raw):,} artículos")
    st.dataframe(raw[["document_title", "publication_year", "publication_title", "document_identifier",
                      "article_citation_count", "query"]], hide_index=True, height=300)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Descargar CSV", csv_bytes(raw), "sota_raw.csv", "text/csv")
    c2.page_link("app_pages/clean.py", label="Siguiente: limpiar →", icon="🧹")
