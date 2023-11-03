import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Paso 1: Carga de datos
books_df = pd.read_csv('archive/Books.csv')
ratings_df = pd.read_csv('archive/Ratings.csv')

# Paso 2: Preprocesamiento de datos (puede incluir la limpieza de datos)

# Paso 3: Sistema de recomendación basado en contenido
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['Book-Title'])

# Cálculo de similitud de coseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función para obtener recomendaciones basadas en contenido
def get_content_based_recommendations(title):
    idx = books_df.index[books_df['Book-Title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Recomendaciones de los 10 libros más similares
    book_indices = [i[0] for i in sim_scores]
    return books_df['Book-Title'].iloc[book_indices]

# Ejemplo de recomendación basada en contenido
recommendations_content_based = get_content_based_recommendations('Classical Mythology')
print("Recomendaciones basadas en contenido:")
print(recommendations_content_based)

# Paso 4: Sistema de recomendación basado en filtrado colaborativo
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# Ejemplo de recomendación basada en filtrado colaborativo
def get_collaborative_filtering_recommendations(user_id, n=10):
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]
    user_unrated_books = books_df[~books_df['ISBN'].isin(user_ratings['ISBN'])]

    user_unrated_books['Predicted-Rating'] = user_unrated_books['ISBN'].apply(
        lambda x: model.predict(user_id, x).est)

    recommended_books = user_unrated_books.sort_values(by='Predicted-Rating', ascending=False)
    return recommended_books[['Book-Title', 'Predicted-Rating']].head(n)

# Ejemplo de recomendación basada en filtrado colaborativo
user_id = 276725
recommendations_collaborative_filtering = get_collaborative_filtering_recommendations(user_id)
print("\nRecomendaciones basadas en filtrado colaborativo:")
print(recommendations_collaborative_filtering)
