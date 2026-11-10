"""SOTA Explorer: state-of-the-art reviews from IEEE Xplore.

Run locally:  streamlit run streamlit_app.py
"""
import streamlit as st

from sota.ieee import IEEEClient, IEEEError

st.set_page_config(page_title="SOTA Explorer", page_icon="📚", layout="wide")

# Small visual polish on top of the theme in .streamlit/config.toml.
st.html("""<style>
h1 {letter-spacing: -0.02em; font-weight: 700;}
[data-testid="stMetric"] {background: #F2F5FB; border: 1px solid #E3E8F2; border-radius: .6rem;
                          padding: .55rem .9rem;}
[data-testid="stMetricValue"] {font-size: clamp(1rem, 1.6vw, 1.5rem); font-weight: 700; color: #1D4ED8;}
.hero {padding: 2.2rem 2rem; border-radius: .9rem; color: #fff; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #0B1A3A 0%, #1D4ED8 70%, #0EA5E9 100%);}
.hero h1 {color: #fff; margin: 0 0 .4rem 0; font-size: 2.3rem;}
.hero p {color: #DCE6FB; font-size: 1.05rem; margin: 0; max-width: 46rem;}
.step {font-size: .92rem; color: #334155;}
.step b {color: #0B1220;}
</style>""")

GALLERY = st.Page("app_pages/gallery.py", title="SOTA publicados", icon=":material/collections_bookmark:")


def home() -> None:
    st.html("""<div class="hero"><h1>SOTA Explorer</h1>
    <p>Construye y explora el estado del arte de cualquier área a partir de IEEE Xplore:
    búsqueda, limpieza asistida y mapas interactivos de temas, autores y tendencias.</p></div>""")

    left, right = st.columns(2, gap="large")
    with left.container(border=True, height="stretch"):
        st.subheader(":material/travel_explore: Explorar SOTA publicados")
        st.write("Revisa estados del arte ya procesados por otras personas. Sin registro ni API key.")
        if st.button("Ver SOTA publicados", type="primary", width="stretch", icon=":material/arrow_forward:"):
            st.switch_page(GALLERY)

    with right.container(border=True, height="stretch"):
        st.subheader(":material/add_circle: Crear un SOTA")
        with st.form("login", border=False):
            key = st.text_input("API key de IEEE Xplore", type="password", placeholder="Pega tu key aquí",
                                help="Obtén una gratis en https://developer.ieee.org. Solo se guarda en la memoria "
                                     "de esta sesión: no se escribe en disco ni en logs y se borra al salir.")
            submitted = st.form_submit_button("Ingresar", type="primary", width="stretch")
        if submitted:
            try:
                IEEEClient(key.strip()).validate()
            except IEEEError as e:
                st.error(str(e))
            else:
                st.session_state.api_key = key.strip()
                st.rerun()
        st.caption(":material/lock: Tu key viaja cifrada (HTTPS) y nunca se guarda.")
        if st.button("Sin key: subir CSV exportados de IEEE Xplore", width="stretch", type="tertiary"):
            st.session_state.api_key = ""
            st.rerun()

    st.write("")
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="step"><b>1 · Buscar.</b> Escribe un tema, mira cuántos artículos hay y descarga '
                'los más relevantes, recientes o citados.</div>', unsafe_allow_html=True)
    s2.markdown('<div class="step"><b>2 · Limpiar.</b> La app agrupa los artículos y te muestra los que no '
                'calzan; decides con un clic y se arma una regla reutilizable.</div>', unsafe_allow_html=True)
    s3.markdown('<div class="step"><b>3 · Explorar y compartir.</b> Temas, mapa, autores y tendencias. '
                'Publícalo para que cualquiera lo vea.</div>', unsafe_allow_html=True)


if "api_key" not in st.session_state:
    pages = [st.Page(home, title="Inicio", icon=":material/home:", default=True), GALLERY,
             st.Page("app_pages/explore.py", title="Explorar", icon=":material/insights:")]
else:
    pages = [
        st.Page("app_pages/search.py", title="1 · Buscar", icon=":material/search:", default=True),
        st.Page("app_pages/clean.py", title="2 · Limpiar", icon=":material/filter_alt:"),
        st.Page("app_pages/explore.py", title="3 · Explorar", icon=":material/insights:"),
        GALLERY,
    ]
    with st.sidebar:
        st.caption("Conectado a IEEE Xplore ✅" if st.session_state.api_key else "Modo sin API key")
        if st.button("Cerrar sesión", width="stretch", icon=":material/logout:"):
            st.session_state.clear()
            st.rerun()

st.navigation(pages).run()
