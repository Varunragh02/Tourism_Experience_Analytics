import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ========== Load Data ========== 
print("📥 Loading cleaned dataset...")
df = pd.read_csv("D:/Guvi/Tourism_Experience_Analytics/Merged_Cleaned.csv")

# ========== Select Features ========== 
features = ['VisitMode', 'Continent', 'Country', 'Region', 'CityName', 'AttractionType', 'Rating']
df = df[features + ['UserId']].copy()

# ========== Handle Categorical Variables ========== 
print("🔤 Encoding categorical features...")
label_encoders = {}
for col in ['Region', 'CityName', 'Continent', 'Country', 'AttractionType','VisitMode']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# One-hot encode 'VisitMode' and 'AttractionTypeId'
df = pd.get_dummies(df, columns=['VisitMode'], drop_first=True)

# ========== Aggregate User Features ========== 
print("📊 Aggregating user-level stats...")
user_agg = df.groupby('UserId').agg(
    avg_rating=('Rating', 'mean'),
    total_visits=('UserId', 'count')
).reset_index()

df = pd.merge(df, user_agg, on='UserId', how='left')

# ========== Normalize Aggregated Features ========== 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Normalized_Rating', 'Normalized_Visits']] = scaler.fit_transform(df[['avg_rating', 'total_visits']])

# ========== Prepare Final Dataset ========== 
df.drop(columns=['UserId'], inplace=True)

X = df.drop(columns=['Rating'])
y = df['Rating']

# ========== Split ========== 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Define Models ========== 
models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.01),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# ========== Cross-Validation ========== 
print("🔁 Performing 5-Fold Cross-Validation...")
results = []
for name, model in models.items():
    # Include R² score in cross-validation scoring
    r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    avg_r2 = np.mean(r2_scores)
    results.append((name, avg_r2))
    print(f"{name} → Avg CV R²: {avg_r2:.4f}")

# ========== Select Best Model ========== 
best_model_name, best_r2 = sorted(results, key=lambda x: x[1], reverse=True)[0]  # Sort by highest R²
print(f"\n🏆 Best Model: {best_model_name} with R²: {best_r2:.4f}")
best_model = models[best_model_name]

# ========== Hyperparameter Tuning ========== 
if best_model_name == "RandomForest":
    print("🔧 Tuning Random Forest...")
    grid = GridSearchCV(best_model, {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }, cv=3, scoring='r2', n_jobs=-1)  # Use R² as scoring
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

elif best_model_name == "GradientBoosting":
    print("🔧 Tuning Gradient Boosting...")
    grid = GridSearchCV(best_model, {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }, cv=3, scoring='r2', n_jobs=-1)  # Use R² as scoring
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

# ========== Final Evaluation ========== 
print("📊 Evaluating final model on test set...")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Manual calculation of RMSE
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Evaluation Metrics
print(f"✅ Test RMSE: {rmse:.4f}")
print(f"✅ Test MSE: {mse:.4f}")
print(f"✅ Test MAE: {mae:.4f}")
print(f"✅ Test R² Score: {r2:.4f}")

# ========== Feature Importance ========== 
if best_model_name in ["RandomForest", "GradientBoosting"]:
    print("\n📊 Feature Importance:")
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    print(importance_df)

# ========== Save Model & Encoders ========== 
joblib.dump(best_model, "D:/Guvi/Tourism_Experience_Analytics/best_model.pkl")
joblib.dump(label_encoders, "D:/Guvi/Tourism_Experience_Analytics/label_encoders.pkl")
joblib.dump(scaler, "D:/Guvi/Tourism_Experience_Analytics/scaler.pkl")
print("💾 Model and encoders saved successfully.")
