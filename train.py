import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("model", exist_ok=True)

data = pd.read_csv('dataset/data.csv')

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
y = data['label']
X = data.drop(['label'], axis=1)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
with open("model/pca.pkl", "wb") as f:
    pickle.dump(pca, f)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, 
                                  min_samples_leaf=5, class_weight='balanced', random_state=42)
rf_model.fit(X_train_pca, y_train)
with open("model/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Predictions
y_pred_rf = rf_model.predict(X_test_pca)

print("\n🔹 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

cv_scores_rf = cross_val_score(rf_model, X_train_pca, y_train, cv=5, scoring='accuracy')
print(f"\n🔹 Random Forest CV Accuracy: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

print("\n✅ Model, PCA, Label Encoder, and Scaler saved successfully.")