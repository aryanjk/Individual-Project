import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (Ensure 'synthetic_amazon_reviews.csv' exists in the same directory)
data = pd.read_csv("synthetic_amazon_reviews.csv")

# Preprocessing: Combine relevant text columns for search filtering
data['combined_features'] = data['summary'] + " " + data['reviewText']

# TF-IDF Vectorization of the combined text
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['combined_features'])

# Function to recommend books based on search filters
def recommend_books(search_query, top_n=5):
    # Vectorize the search query
    query_vector = vectorizer.transform([search_query])
    
    # Calculate cosine similarity between the search query and all books
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get the top N books with the highest similarity scores
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Retrieve and return the recommended books
    recommendations = data.iloc[top_indices][['asin', 'summary', 'overall']]
    return recommendations

# Example usage
if __name__ == "__main__":
    search_filter = "mystery novel detective"  # Example search query
    top_recommendations = recommend_books(search_filter, top_n=5)
    print("Top Book Recommendations:\n")
    print(top_recommendations)