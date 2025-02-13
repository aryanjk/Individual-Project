import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load the dataset
data = pd.read_csv("synthetic_amazon_reviews.csv")

# Preprocessing: Combine relevant text columns for search filtering
data['combined_features'] = data['summary'] + " " + data['reviewText']

# Drop missing values for required columns
data = data.dropna(subset=['reviewerID', 'asin', 'overall'])

# Ensure data types are correct
data['reviewerID'] = data['reviewerID'].astype(str)
data['asin'] = data['asin'].astype(str)
data['overall'] = data['overall'].astype(float)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Content-Based Filtering: Save the TF-IDF vectorizer and dataset for reuse
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('book_data.pkl', 'wb') as f:
    pickle.dump(data, f)

# Collaborative Filtering: Create a user-book matrix
user_book_matrix = data.pivot_table(index='reviewerID', columns='asin', values='overall', aggfunc='mean').fillna(0)

# Fit NearestNeighbors model for user-based filtering
neighbors_model = NearestNeighbors(metric='cosine', algorithm='brute')
neighbors_model.fit(user_book_matrix)

# Save the collaborative filtering model
with open('neighbors_model.pkl', 'wb') as f:
    pickle.dump(neighbors_model, f)

# Flask app
app = Flask(__name__)

# Load the pre-saved models and data
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('book_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('neighbors_model.pkl', 'rb') as f:
    neighbors_model = pickle.load(f)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# API route for book recommendation
@app.route('/recommend', methods=['POST'])
# def recommend():
#     search_query = request.form.get('search_query')
#     top_n = int(request.form.get('top_n', 5))

#     # Content-Based Filtering: Vectorize the search query
#     query_vector = vectorizer.transform([search_query])
#     cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

#     # Get top N content-based recommendations
#     top_content_indices = cosine_similarities.argsort()[-top_n:][::-1]
#     content_recommendations = data.iloc[top_content_indices][['asin', 'summary', 'overall']]

#     # Collaborative Filtering: Find similar users based on reviewerID
#     reviewer_id = request.form.get('reviewer_id')
#     if reviewer_id in user_book_matrix.index:
#         distances, indices = neighbors_model.kneighbors(
#             [user_book_matrix.loc[reviewer_id]], n_neighbors=top_n + 1
#         )
#         collaborative_recommendations = data[data['asin'].isin(
#             user_book_matrix.columns[indices.flatten()[1:]]
#         )][['asin', 'summary', 'overall']]
#     else:
#         collaborative_recommendations = pd.DataFrame(columns=['asin', 'summary', 'overall'])

#     # Combine and deduplicate results from both methods
#     combined_recommendations = pd.concat(
#         [content_recommendations, collaborative_recommendations]
#     ).drop_duplicates(subset='asin').head(top_n)

#     # Convert recommendations to a list of dictionaries
#     result = combined_recommendations.to_dict(orient='records')
#     return jsonify(result)

@app.route('/recommend', methods=['POST'])
def recommend():
    search_query = request.form.get('search_query')
    top_n = int(request.form.get('top_n', 5))

    # Debugging: Check if the search query is received
    print("Received Search Query:", search_query)
    
    if not search_query or search_query.strip() == "":
        return jsonify({"error": "Empty search query"}), 400

    # Vectorize the search query using TF-IDF
    query_vector = vectorizer.transform([search_query])
    
    # Debugging: Check if TF-IDF vectorization is working
    print("Query Vector Shape:", query_vector.shape)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Sort indices based on similarity scores
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Get content-based recommendations
    content_recommendations = data.iloc[top_indices][['asin', 'summary', 'overall']]
    
    # Collaborative Filtering: Find similar users (if reviewerID is provided)
    reviewer_id = request.form.get('reviewer_id')
    collaborative_recommendations = pd.DataFrame(columns=['asin', 'summary', 'overall'])

    if reviewer_id and reviewer_id in user_book_matrix.index:
        distances, indices = neighbors_model.kneighbors([user_book_matrix.loc[reviewer_id]], n_neighbors=top_n + 1)
        collaborative_recommendations = data[data['asin'].isin(user_book_matrix.columns[indices.flatten()[1:]])][['asin', 'summary', 'overall']]
    
    # Debugging: Check if collaborative filtering contributes unique recommendations
    print("Content-Based Recommendations:", content_recommendations)
    print("Collaborative Filtering Recommendations:", collaborative_recommendations)

    # Merge & deduplicate recommendations
    combined_recommendations = pd.concat([content_recommendations, collaborative_recommendations]).drop_duplicates(subset='asin').head(top_n)

    return jsonify(combined_recommendations.to_dict(orient='records'))


if __name__ == "__main__":
    app.run(debug=True)
