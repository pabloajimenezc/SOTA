import os
import sys
import subprocess
from pathlib import Path

def find_existing_venv(start_path):
    current = start_path.resolve()
    while current != current.parent:
        for name in ['.venv', 'venv']:
            candidate = current / name
            if (candidate / ('Scripts' if os.name == 'nt' else 'bin') / 'python').exists():
                return candidate
        current = current.parent
    return None

def find_project_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent

def create_venv(venv_path):
    print(f'Creando entorno virtual en: {venv_path}')
    subprocess.run([sys.executable, '-m', 'venv', str(venv_path)])
    return venv_path

# 1. Buscar entorno existente o crear uno
script_dir = Path(__file__).resolve().parent
venv = find_existing_venv(script_dir)

if venv:
    print(f'Entorno virtual encontrado en: {venv}')
else:
    project_root = find_project_root()
    venv = project_root / '.venv'
    venv = create_venv(venv)

# 2. Detectar ejecutable de Python dentro del venv
python_exec = (
    venv / 'Scripts' / 'python.exe' if os.name == 'nt'
    else venv / 'bin' / 'python'
)

# 3. Instalar paquetes
libraries = [
    'Levenshtein',
    'certifi',
    'fastcluster',
    'gensim',
    'graphviz',
    'ipywidgets',
    'jupyter',
    'kaleido',
    'nbformat',
    'networkx',
    'nltk',
    'plotly',
    'pyDOE',
    'pycurl',
    'pydot',
    'seaborn',
    'spacy',
    'umap-learn',
    'wordcloud',
    'yellowbrick',
    'numpy==1.26.4'
]

print(f'Instalando {len(libraries)} paquetes...')
for package in libraries:
    subprocess.check_call([str(python_exec), '-m', 'pip', 'install', package])

# 4. Descargar recursos
subprocess.run([str(python_exec), '-m', 'spacy', 'download', 'en_core_web_sm'])
subprocess.run([str(python_exec), '-c', 'import nltk; nltk.download("punkt"); nltk.download("punkt_tab"); nltk.download("wordnet"); nltk.download("stopwords")'])

# 5. Guardar requirements.txt
project_root = find_project_root()
req_file = project_root / 'requirements.txt'
print(f'Guardando dependencias en: {req_file}')
with open(req_file, 'w') as file:
    subprocess.run([str(python_exec), '-m', 'pip', 'freeze'], stdout=file)