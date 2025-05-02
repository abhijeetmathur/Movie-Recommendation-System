import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load and preprocess data
movies_data = pd.read_csv('/content/movies.csv')
movies_data.fillna('', inplace=True)
movies_data.reset_index(drop=True, inplace=True)

# Combine selected text features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
movies_data['combined'] = movies_data[selected_features].agg(' '.join, axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(movies_data['combined']).toarray()

# Create weak labels using cosine similarity
pairs = []
labels = []

for i, j in combinations(range(len(tfidf_matrix)), 2):
    sim = cosine_similarity([tfidf_matrix[i]], [tfidf_matrix[j]])[0][0]
    label = 1 if sim > 0.8 else 0  # Weak supervision
    pairs.append((tfidf_matrix[i], tfidf_matrix[j]))
    labels.append(label)

X1 = np.array([pair[0] for pair in pairs])
X2 = np.array([pair[1] for pair in pairs])
y = np.array(labels)

# Split into training and test sets
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Define base network to encode each movie vector
def base_network(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    return Model(inputs=inp, outputs=x)

input_shape = X1.shape[1]
base_net = base_network(input_shape)

# Inputs
input_a = Input(shape=(input_shape,))
input_b = Input(shape=(input_shape,))

# Encoded vectors
encoded_a = base_net(input_a)
encoded_b = base_net(input_b)

# Merge the two vectors
merged = Concatenate()([encoded_a, encoded_b])
x = Dense(128, activation='relu')(merged)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Final model
model = Model(inputs=[input_a, input_b], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X1_train, X2_train], y_train, validation_data=([X1_test, X2_test], y_test), epochs=5, batch_size=128)

# Prediction function
def recommend_movies_tf(movie_name):
    titles = movies_data['title'].tolist()
    matches = get_close_matches(movie_name, titles)
    if not matches:
        print("Movie not found.")
        return

    selected = matches[0]
    idx = movies_data[movies_data['title'] == selected].index[0]
    movie_vec = tfidf_matrix[idx]

    similarities = []
    for i in range(len(tfidf_matrix)):
        if i == idx:
            continue
        score = model.predict([[movie_vec], [tfidf_matrix[i]]])[0][0]
        similarities.append((i, score))

    top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    print(f"\nRecommendations based on '{selected}':\n")
    for i, (idx, score) in enumerate(top_similar, 1):
        print(f"{i}. {movies_data.iloc[idx]['title']} (Similarity Score: {score:.2f})")

# Get user input
user_input = input("Enter your favourite movie name: ")
recommend_movies_tf(user_input)
