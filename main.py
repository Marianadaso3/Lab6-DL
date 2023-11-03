import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Carga de datos
books_df = pd.read_csv('archive/Books.csv')
users_df = pd.read_csv('archive/Users.csv')
ratings_df = pd.read_csv('archive/Ratings.csv')

# Preprocesamiento de datos
# Aquí deberás realizar la limpieza y preprocesamiento de tus datos, como manejo de valores faltantes, codificación de variables categóricas, etc.

# Construcción de un sistema de recomendación basado en contenido
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['Book-Title'])

# Cálculo de similitud de coseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función para obtener recomendaciones basadas en contenido
def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    idx = books_df.index[books_df['Book-Title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Recomendaciones de los 10 libros más similares
    book_indices = [i[0] for i in sim_scores]
    return books_df['Book-Title'].iloc[book_indices]

# Ejemplo de recomendación basada en contenido
recommendations = get_content_based_recommendations('Classical Mythology')
print(recommendations)

# Implementación del filtrado colaborativo y evaluación
# Aquí deberás implementar el filtrado colaborativo y evaluar su rendimiento utilizando métricas como RMSE o MAE.
