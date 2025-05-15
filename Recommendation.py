import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
from scipy.sparse import csr_matrix

# --- Parameters ---
top_n = 10         # Number of recommendations
alpha = 0.6        # Weight for collaborative filtering (1 - alpha for content-based)

# --- Load Preprocessed Data ---
@st.cache_data
def load_data():
    # Load and preprocess the data (df)
    df = pd.read_csv("D:/Guvi/Tourism_Experience_Analytics/Merged_Cleaned.csv")
    df = df[['UserId', 'AttractionId', 'Attraction', 'AttractionType', 'CityName', 'Country', 'Rating']].dropna()
    
    # Content-based filtering data (non-cached, computed on-demand)
    df_content = df.drop_duplicates(subset=['AttractionId']).copy()
    df_content['CombinedFeatures'] = (
        df_content['Attraction'].astype(str) + ' ' +
        df_content['AttractionType'].astype(str) + ' ' +
        df_content['CityName'].astype(str)
    )
    
    return df, df_content

df, df_content = load_data()

# --- Functions to Compute Similarities (on-demand, not cached) ---
def get_collab_similarity(df):
    # Collaborative Filtering (User-Item Matrix)
    user_item_matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix)  # Use sparse matrix to reduce memory usage
    collab_similarity = cosine_similarity(sparse_matrix)
    
    # Convert to DataFrame and set the correct index and columns
    collab_sim_df = pd.DataFrame(collab_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    return collab_sim_df

def get_content_similarity(df_content):
    # Content-Based Filtering
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit number of features for memory efficiency
    tfidf_matrix = tfidf.fit_transform(df_content['CombinedFeatures'])
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Convert to DataFrame for easy access
    content_sim_df = pd.DataFrame(content_similarity, index=df_content['AttractionId'], columns=df_content['AttractionId'])
    
    return content_sim_df

# --- Recommendation Function ---
def recommend_attractions(user_id, top_n=10, alpha=0.5):
    # Compute similarities dynamically
    collab_sim_df = get_collab_similarity(df)
    content_sim_df = get_content_similarity(df_content)

    if user_id not in collab_sim_df.index:
        st.error(f"❌ User {user_id} not found.")
        return pd.DataFrame()

    # Collaborative Filtering Scores
    similar_users = pd.Series(collab_sim_df[user_id], index=collab_sim_df.index).drop(index=user_id).sort_values(ascending=False)
    top_sim_users = similar_users.head(10).index
    similar_users_avg = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').loc[top_sim_users].mean()

    # Attractions already rated by user
    rated_attractions = df.pivot_table(index='UserId', columns='AttractionId', values='Rating').loc[user_id]
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

# --- Streamlit UI ---
st.title("Tourism Experience Recommendations")

# Get available user IDs
available_user_ids = sorted(df['UserId'].unique())

# Dropdown to select valid user ID
user_id_input = st.selectbox("Select a UserId", available_user_ids)

# Generate recommendations when button is clicked
if st.button("Get Recommendations"):
    recommendations = recommend_attractions(user_id_input, top_n=top_n, alpha=alpha)

    
    if not recommendations.empty:
        st.subheader(f"Top {top_n} Recommendations for User {user_id_input}:")
        st.write(recommendations)

        # Download recommendations as CSV
        csv = recommendations.to_csv(index=False)
        st.download_button("Download Recommendations as CSV", csv, f"user_{user_id_input}_recommendations.csv", "text/csv")

        # Save recommendations to SQLite
        conn = sqlite3.connect("tourism_recommendations.db")
        recommendations.to_sql(f"user_{user_id_input}_recommendations", conn, if_exists='replace', index=False)
        conn.close()
        st.success(f"Recommendations stored in SQLite: table user_{user_id_input}_recommendations")
    else:
        st.warning("No recommendations found for the given UserId.")