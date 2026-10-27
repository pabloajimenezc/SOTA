# SOTA Explorer

App web para construir el estado del arte desde **IEEE Xplore**: búsqueda por API, limpieza asistida por
clusters con reglas reutilizables y visualización interactiva.

## Flujo

1. **Buscar**: ingresa tu tema (admite la sintaxis booleana de IEEE), filtra por tipo y años, mira cuántos
   artículos hay y descarga los N primeros ordenados por relevancia, más recientes, más antiguos, más
   citados o título. También acepta CSV exportados manualmente desde IEEE Xplore.
2. **Limpiar**: clusters ordenados del más distinto al resto, revisión de abstracts uno a uno (los más
   atípicos primero) y términos sugeridos. Cada decisión queda en una **regla** (`.json`) que puedes
   descargar y reutilizar. Ejemplo: `rules/electromovilidad_maritima.json`.
3. **Explorar**: publicaciones por año, jerarquía y evolución de tópicos, mapa UMAP con selección,
   búsqueda semántica con perfil por tópicos, términos emergentes, autores, fuentes y países, y descarga
   del CSV con tópicos.

4. **SOTA publicados**: desde la página de inicio cualquiera puede *Revisar otros SOTA* sin API key. Quien
   tiene sesión puede publicar el suyo desde *Explorar → Publicar*. Se guardan los artículos limpios, el mapa,
   la regla y el número de tópicos en un Parquet comprimido (~5 MB por cada 10k artículos) en la rama
   `sotas` del repo (`index.json` + `sotas/<nombre>.parquet`). Solo la misma API key puede volver a
   publicar con un nombre ya usado.

## Seguridad de la API key

- Cada usuario ingresa **su propia** key de [developer.ieee.org](https://developer.ieee.org).
- Se guarda solo en la memoria de su sesión (`st.session_state`). No se escribe en disco, no se cachea
  entre usuarios, no aparece en logs ni en mensajes de error, y se borra al cerrar sesión o la pestaña.
- Viaja cifrada por HTTPS (navegador → app → IEEE).

## Uso local

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # Linux/Mac: .venv/bin/pip
.venv/Scripts/streamlit run streamlit_app.py
.venv/Scripts/python -m pytest tests
```

## Publicar gratis (Streamlit Community Cloud)

1. Sube el repositorio a GitHub (la carpeta `data/` está ignorada y no se sube).
2. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub → **Create app** →
   elige el repo, rama `main` y archivo `streamlit_app.py`.
3. Listo: obtendrás una URL `https://<nombre>.streamlit.app` accesible desde cualquier dispositivo.
4. Para habilitar **Publicar**: crea un token *fine-grained* en GitHub (Settings → Developer settings →
   Personal access tokens) con acceso solo a este repo y permiso *Contents: Read and write*. Después,
   en la app de Streamlit Cloud ve a **Settings → Secrets** y pega:

   ```toml
   [github]
   repo = "pabloajimenezc/SOTA"
   branch = "sotas"
   token = "github_pat_..."
   ```

   Sin token, la galería funciona igual en modo solo lectura (si el repo es público).

Límites del plan gratis: ~1 GB de RAM (suficiente para ~20k artículos) y la app se duerme tras unos días sin
uso (se despierta en ~30 s al abrirla).

## Estructura

```
streamlit_app.py        login y navegación
app_pages/              páginas (buscar, limpiar, explorar) y estado compartido
sota/
  ieee.py               cliente de la API de IEEE Xplore (requests)
  data.py               normalización de API/CSV a las columnas del proyecto, deduplicación
  text.py               preprocesamiento, TF-IDF + LSA, clusters, búsqueda semántica, UMAP
  cleaning.py           reglas de limpieza (JSON)
  store.py              SOTA publicados (Parquet + index.json en la rama `sotas`)
  viz.py                gráficos Plotly
rules/                  reglas de ejemplo
tests/                  tests (sin llamadas reales a la API)
notebooks/              análisis original (usa requirements-notebooks.txt)
```

Para usar el paquete desde los notebooks: `sys.path.append('..')` y `from sota import ieee, data, text`.
