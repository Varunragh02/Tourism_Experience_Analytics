import streamlit as st
import os

# Set up the main page configuration
st.set_page_config(page_title="Tourism Dashboard", layout="wide")

# Title of the app
st.title("🌍 Tourism Experience App")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_choice = st.sidebar.radio("Choose an App:", ["🏠 Home", "🎯 Tourism Rating Predictor", "🧭 Experience Classifier", "🛫 Recommendation"])

# Main content based on app choice
if app_choice == "🏠 Home":
    st.markdown("""
    Welcome to the **Tourism Experience App**! 

    You can predict tourism ratings or classify experiences based on the features provided.
    Select the app you want to explore from the sidebar on the left.
    """)
    
elif app_choice == "🎯 Tourism Rating Predictor":
    st.markdown("Running **Tourism Rating Predictor App**...")
    os.system("streamlit run rating_predictor.py")

elif app_choice == "🧭 Experience Classifier":
    st.markdown("Running **Experience Classifier App**...")
    os.system("streamlit run experience_classifier.py")
    # Similarly, you can import and call the content of your 'experience_classifier.py' script
    # from experience_classifier import run_classifier
    # run_classifier()
elif app_choice == "🛫 Recommendation":
    st.markdown("Running **Recommendation App**...")
    os.system("streamlit run Recommendation.py")
    # Similarly, you can import and call the content of your 'recommendation.py' script
    # from recommendation import run_recommendation
    # run_recommendation()

