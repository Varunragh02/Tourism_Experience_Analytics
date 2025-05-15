import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder
import joblib

# ========== Load Data ========== 
print("📥 Loading cleaned dataset...")
df = pd.read_csv("D:/Guvi/Tourism_Experience_Analytics/Merged_Cleaned.csv")

# ========== Select Features and Target ========== 
features = ['Continent', 'Country', 'Region', 'CityName', 'AttractionType', 'VisitYear', 'VisitMonth']
df = df[features + ['VisitMode']].copy()
X = df[features]
y = df['VisitMode']
print("✅ Features and target separated.")

# ========== K-Fold Cross Validation ========== 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
roc_aucs = []

fold = 1
for train_index, test_index in skf.split(X, y):
    print(f"\n🔁 Fold {fold}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Target Encoding
    encoder = TargetEncoder()
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_encoded, y_train)

    # Train Random Forest
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_balanced, y_train_balanced)

    # Predict and Evaluate
    y_pred = rf_model.predict(X_test_encoded)
    y_proba = rf_model.predict_proba(X_test_encoded)

    acc = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    except ValueError:
        roc_auc = np.nan  # fallback if only one class is predicted
    
    accuracies.append(acc)
    roc_aucs.append(roc_auc)

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🧮 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"🎯 Accuracy: {acc:.4f}")
    print(f"🧪 ROC AUC: {roc_auc:.4f}")
    
    fold += 1

# ========== Summary ==========
print("\n📈 Cross-Validation Summary:")
print(f"✅ Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"✅ Mean ROC AUC: {np.nanmean(roc_aucs):.4f} ± {np.nanstd(roc_aucs):.4f}")

# ========== Final Model Training on Full Data ==========
print("\n🧠 Training final model on full dataset...")

# Encode full data
final_encoder = TargetEncoder()
X_encoded = final_encoder.fit_transform(X, y)

# SMOTE
final_smote = SMOTE(random_state=42)
X_balanced, y_balanced = final_smote.fit_resample(X_encoded, y)

# Train final model
final_model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_balanced, y_balanced)

# Feature Importance Plot
importances = final_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("✅ Feature Importance:")
print(feature_imp_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['Feature'][:10], feature_imp_df['Importance'][:10], color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features (Final Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Save final model and encoder
joblib.dump(final_model, "best_classification_model.pkl")
joblib.dump(final_encoder, "target_encoder.pkl")
print("✅ Final model and encoder saved.")
