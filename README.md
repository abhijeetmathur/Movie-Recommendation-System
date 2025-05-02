🎬 Movie Recommendation System (Deep Learning + TF-IDF + TensorFlow)
This project builds a content-based movie recommendation system using TF-IDF vectorization and a deep learning model built with TensorFlow/Keras. It uses a Siamese-like architecture to predict similarity between pairs of movies based on features such as genre, cast, keywords, tagline, and director.

📁 Dataset
The system uses a dataset containing information about movies, including the following relevant fields:

title
genres
keywords
tagline
cast
director

These features are combined into a single text field and vectorized using TF-IDF to compute similarities between movies.

🧠 Machine Learning Model
Architecture: Dual-input neural network (Siamese-style) with:

Dense layers and dropout for regularization

Feature vector concatenation

Sigmoid output to predict similarity

Training Labels: Weak supervision based on cosine similarity of TF-IDF vectors (> 0.8 treated as similar).

🚀 How It Works
Preprocessing:

Combine features: genres, keywords, tagline, cast, director.

Use TfidfVectorizer to convert combined text to numerical vectors.

Generate pairwise data points with similarity labels.

Model Training:

Train the deep learning model to predict similarity between movie pairs.

Recommendation:

Given a movie title, the system finds the closest match and predicts similarity scores with all other movies.

Top 10 most similar movies are recommended.

🛠️ Setup Instructions
1. Install Dependencies:
pip install pandas numpy tensorflow scikit-learn

2. Run the Script:
python movie_recommender.py


3. Provide Input:
You’ll be prompted to enter your favorite movie title. For example:

Enter your favourite movie name: Batman

The script will output the top 10 most similar movies based on learned similarities.

📌 Example Output
Recommendations based on 'The Dark Knight':
1. Batman Begins (Similarity Score: 0.93)
2. The Dark Knight Rises (Similarity Score: 0.89)
3. Man of Steel (Similarity Score: 0.85)

📂 File Structure
├── Movie_Recommender.py     # Main Python script
├── Movies.csv               # Dataset file
└── README.md                # This file
