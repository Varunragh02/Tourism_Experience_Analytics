import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sqlite3

# Parameters
top_n = 10
alpha = 0.6

# Load data
df = pd.read_csv("D:/Guvi/Tourism_Experience_Analytics/Merged_Cleaned.csv")
df = df[['UserId', 'AttractionId', 'Attraction', 'AttractionType', 'CityName', 'Country', 'Rating']].dropna()

# Create user-item matrix
user_item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
print(f"User-Item matrix shape: {user_item_matrix.shape}")

# Convert to sparse matrix for memory efficiency
sparse_user_item = csr_matrix(user_item_matrix.values)

# Fit NearestNeighbors model for collaborative filtering (user-user similarity)
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
knn_model.fit(sparse_user_item)
print("✅ Fitted NearestNeighbors model for collaborative filtering")

# Prepare content-based similarity matrix
df_content = df.drop_duplicates(subset=['AttractionId']).copy()
df_content['CombinedFeatures'] = (
    df_content['Attraction'].astype(str) + ' ' +
    df_content['AttractionType'].astype(str) + ' ' +
    df_content['CityName'].astype(str)
)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_content['CombinedFeatures'])
content_similarity = (tfidf_matrix * tfidf_matrix.T).toarray()
content_sim_df = pd.DataFrame(content_similarity, index=df_content['AttractionId'], columns=df_content['AttractionId'])
print("✅ Calculated content-based similarity matrix")

# Save models (optional)
joblib.dump((user_item_matrix, knn_model), 'collaborative_model.pkl')
joblib.dump((df_content, tfidf, content_sim_df), 'content_model.pkl')

def recommend_attractions(user_id, top_n=10, alpha=0.6):
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found.")
        return pd.DataFrame()

    user_idx = user_item_matrix.index.get_loc(user_id)

    # Find top 10 similar users using KNN
    distances, indices = knn_model.kneighbors(sparse_user_item[user_idx], n_neighbors=top_n + 1)
    distances = distances.flatten()
    indices = indices.flatten()

    # Exclude the user itself from neighbors
    neighbors_idx = indices[1:]
    neighbors_distances = distances[1:]

    # Similarity scores (1 - cosine distance)
    similarity_scores = 1 - neighbors_distances

    # Get ratings of neighbors
    neighbors_ratings = user_item_matrix.iloc[neighbors_idx]

    # Weighted average ratings
    weighted_ratings = neighbors_ratings.T.dot(similarity_scores) / similarity_scores.sum()

    # Exclude attractions already rated by user
    user_ratings = user_item_matrix.iloc[user_idx]
    unrated_mask = user_ratings == 0
    weighted_ratings = weighted_ratings[unrated_mask]

    # Content-based scores for these attractions
    rated_attractions = user_ratings[user_ratings > 0].index.intersection(content_sim_df.columns)
    content_scores = pd.Series(0, index=weighted_ratings.index)
    for attraction_id in weighted_ratings.index:
        if attraction_id in content_sim_df.columns and not rated_attractions.empty:
            content_scores[attraction_id] = content_sim_df.loc[attraction_id, rated_attractions].mean()

    # Hybrid score combining collaborative and content-based
    hybrid_scores = alpha * weighted_ratings + (1 - alpha) * content_scores

    # Top N recommendations
    top_recommendations = hybrid_scores.sort_values(ascending=False).head(top_n)

    # Add metadata
    result_df = df_content[df_content['AttractionId'].isin(top_recommendations.index)][
        ['AttractionId', 'Attraction', 'AttractionType', 'CityName', 'Country']
    ].copy()
    result_df['Score'] = result_df['AttractionId'].map(top_recommendations)

    return result_df.sort_values(by='Score', ascending=False)

# Example usage
user_id = 87100
recommendations = recommend_attractions(user_id, top_n=top_n, alpha=alpha)

if not recommendations.empty:
    print(recommendations)

    # Save to CSV
    recommendations.to_csv(f"user_{user_id}_recommendations.csv", index=False)

    # Save to SQLite
    conn = sqlite3.connect("tourism_recommendations.db")
    recommendations.to_sql(f"user_{user_id}_recommendations", conn, if_exists='replace', index=False)
    conn.close()
else:
    print("No recommendations generated.")
