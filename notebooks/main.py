import pandas as pd
import matplotlib.pyplot as plt
import re, string, joblib, umap
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram, fcluster
from wordcloud import WordCloud
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Importar para graficar en 3D

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Inicializar lemmatizer y stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Función para preprocesar texto
def preprocess_text(text, words2remove):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in words2remove]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Función para generar nube de palabras
def word_cloud(text):
    word_freq = Counter(text.split(' '))
    wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# Función principal para ejecutar el análisis
def run_analysis(query_text):
    # Cargar datos
    data = pd.read_csv('D:/Users/Pablo/Desktop/Tesis/SOTA/data/input/api/data_filtered.csv')
    df = pd.read_csv('D:/Users/Pablo/Desktop/Tesis/SOTA/data/input/api/data_cleaned.csv')
    df_vectorized = pd.read_csv('D:/Users/Pablo/Desktop/Tesis/SOTA/data/output/df_vectorized.csv')
    X = df_vectorized.values
    words2remove = []
    vectorizer = joblib.load('D:/Users/Pablo/Desktop/Tesis/SOTA/models/vectorizer.pkl')
    words = vectorizer.get_feature_names_out()

    # Preprocesar texto de consulta
    query_text = [preprocess_text(query_text, words2remove)]
    query_text_vec = vectorizer.transform(query_text)

    # Calcular KNNs
    n_neighbors = 100
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(X)
    knn_distances, knn_idxs = knn.kneighbors(query_text_vec.reshape(1, -1))

    # Mostrar resultados en la interfaz
    result_text.delete(1.0, tk.END)
    for i, knn_idx in enumerate(knn_idxs[0]):
        result_text.insert(tk.END, f"Neighbor {i+1}:\n")
        result_text.insert(tk.END, f"Title: {data['title'][knn_idx]}\n")
        result_text.insert(tk.END, f"Abstract: {data['abstract'][knn_idx]}\n")
        result_text.insert(tk.END, "-"*40 + "\n")

    # UMAP 3D
    reducer = umap.UMAP(n_components=3)
    X_umap_3d = reducer.fit_transform(df_vectorized)
    df['umap3d_1'] = X_umap_3d[:, 0]
    df['umap3d_2'] = X_umap_3d[:, 1]
    df['umap3d_3'] = X_umap_3d[:, 2]

    # Graficar UMAP 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['umap3d_1'], df['umap3d_2'], df['umap3d_3'], c='blue', alpha=0.5)
    ax.scatter(df.loc[knn_idxs[0], 'umap3d_1'], df.loc[knn_idxs[0], 'umap3d_2'], df.loc[knn_idxs[0], 'umap3d_3'], c='red', alpha=1)
    ax.set_title("UMAP 3D")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

    # Mostrar gráfico en la interfaz
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Análisis de Texto")

# Campo de entrada para el texto de consulta
query_label = tk.Label(root, text="Ingrese el texto de consulta:")
query_label.pack()
query_entry = tk.Entry(root, width=50)
query_entry.pack()

# Botón para ejecutar el análisis
analyze_button = tk.Button(root, text="Analizar", command=lambda: run_analysis(query_entry.get()))
analyze_button.pack()

# Área de texto para mostrar resultados
result_text = tk.Text(root, height=20, width=80)
result_text.pack()

# Ejecutar la interfaz
root.mainloop()