import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# ========== Load Data ========== 
print("📥 Loading cleaned dataset...")
df = pd.read_csv("D:/Guvi/Tourism_Experience_Analytics/Merged_Cleaned.csv")

# ========== Select Features ========== 
features = ['Continent', 'Country', 'Region', 'CityName', 'AttractionType', 'VisitYear', 'VisitMonth']
df = df[features + ['VisitMode']].copy()

# ========== Separate X and y ========== 
X = df.drop(columns=['VisitMode'])
y = df['VisitMode']

# ========== Encode Categorical Features Efficiently ========== 
# Use pd.get_dummies to encode categorical columns
X_encoded = pd.get_dummies(X, drop_first=True)

# If performance warning persists, you can try explicitly defragmenting with a copy:
X_encoded = X_encoded.copy()

# STEP 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

# STEP 4: Train RandomForest with class weights
rf_model = RandomForestClassifier(
    class_weight='balanced',  # handles class imbalance
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# STEP 5: Predict and Evaluate
y_pred = rf_model.predict(X_test)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# STEP 6: Feature Importance Plot
importances = rf_model.feature_importances_
feat_names = X_train.columns

feature_imp_df = pd.DataFrame({
    'Feature': feat_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
# Save feature importance
print("Feature Importance:")
print(feature_imp_df)

# Plot Top 10 Features
plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['Feature'][:10], feature_imp_df['Importance'][:10], color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(rf_model, "best_classification_model.pkl")
print("✅ Model saved as 'best_classification_model.pkl'")
