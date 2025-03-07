import sys
import subprocess

# pip install -r requirements

# Install packages
library = [
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
    'yellowbrick'
]
for package in library:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Download language models and utils
import spacy, nltk
spacy.cli.download('en_core_web_sm')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Freeze to requirements.txt
with open('requirements.txt', 'w') as file:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=file)
