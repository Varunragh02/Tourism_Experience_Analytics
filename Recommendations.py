import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sqlite3

# --- Parameters ---
top_n = 10         # Number of recommendations
alpha = 0.6        # Weight for collaborative filtering (1 - alpha for content-based)

# --- Load Data ---
df = pd.read_csv("D:/Guvi/Tourism_Experience_Analytics/Merged_Cleaned.csv")
df = df[['UserId', 'AttractionId', 'Attraction', 'AttractionType', 'CityName', 'Country', 'Rating']].dropna()

# --- Collaborative Filtering (User-Item Matrix) ---
user_item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
collab_similarity = cosine_similarity(user_item_matrix)
collab_sim_df = pd.DataFrame(collab_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# --- Content-Based Filtering ---
df_content = df.drop_duplicates(subset=['AttractionId']).copy()
df_content['CombinedFeatures'] = (
    df_content['Attraction'].astype(str) + ' ' +
    df_content['AttractionType'].astype(str) + ' ' +
    df_content['CityName'].astype(str)
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_content['CombinedFeatures'])
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
content_sim_df = pd.DataFrame(content_similarity, index=df_content['AttractionId'], columns=df_content['AttractionId'])

# --- Save Models ---
joblib.dump((user_item_matrix, collab_sim_df), 'collaborative_model.pkl')
joblib.dump((df_content, tfidf, content_sim_df), 'content_model.pkl')
print("✅ Models saved.")

# --- Recommendation Function ---
def recommend_attractions(user_id, top_n=10, alpha=0.5):
    if user_id not in user_item_matrix.index:
        print(f"❌ User {user_id} not found.")
        return pd.DataFrame()

    # Collaborative Filtering Scores
    similar_users = collab_sim_df[user_id].drop(index=user_id).sort_values(ascending=False)
    top_sim_users = similar_users.head(10).index
    similar_users_avg = user_item_matrix.loc[top_sim_users].mean()

    # Attractions already rated by user
    rated_attractions = user_item_matrix.loc[user_id]
    rated_attractions = rated_attractions[rated_attractions > 0].index.tolist()

    # Filter out already rated attractions
    similar_users_avg = similar_users_avg.drop(labels=rated_attractions, errors='ignore')

    # Content-Based Scores
    content_scores = pd.Series(0, index=similar_users_avg.index)
    for aid in similar_users_avg.index:
        if aid in content_sim_df.columns:
            sim_scores = content_sim_df.loc[aid, rated_attractions].mean()
            content_scores[aid] = sim_scores

    # Combine Hybrid Scores
    hybrid_scores = alpha * similar_users_avg + (1 - alpha) * content_scores
    top_recommendations = hybrid_scores.sort_values(ascending=False).head(top_n)

    # Add metadata
    result_df = df_content[df_content['AttractionId'].isin(top_recommendations.index)][
        ['AttractionId', 'Attraction', 'AttractionType', 'CityName', 'Country']
    ].copy()
    result_df['Score'] = result_df['AttractionId'].map(top_recommendations)

    return result_df.sort_values(by='Score', ascending=False)

# --- Generate & Save Recommendations ---
user_id = 87100  # Change as needed
recommendations = recommend_attractions(user_id, top_n=top_n, alpha=alpha)

if not recommendations.empty:
    print(f"\n🎯 Top {top_n} Recommendations for User {user_id}:\n")
    print(recommendations)

    # Save to CSV
    recommendations.to_csv(f"user_{user_id}_recommendations.csv", index=False)
    print(f"📁 Saved to user_{user_id}_recommendations.csv")

    # Save to SQLite
    conn = sqlite3.connect("tourism_recommendations.db")
    recommendations.to_sql(f"user_{user_id}_recommendations", conn, if_exists='replace', index=False)
    conn.close()
    print(f"🗃️ Stored in SQLite: table user_{user_id}_recommendations")
