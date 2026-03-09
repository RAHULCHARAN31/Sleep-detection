"""
Sleep Detection from Accelerometer Data
Author: Rahul Charan Erigirala
Description: Classifies sleep vs. wake states from wrist accelerometer signals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE SAMPLE DATA (simulates accelerometer data)
# ─────────────────────────────────────────────
np.random.seed(42)
n_samples = 2000

# Simulate anglez and enmo features
anglez_sleep = np.random.normal(-20, 10, n_samples // 2)
anglez_wake  = np.random.normal(10, 20, n_samples // 2)
anglez = np.concatenate([anglez_sleep, anglez_wake])

enmo_sleep = np.random.normal(0.02, 0.01, n_samples // 2)
enmo_wake  = np.random.normal(0.15, 0.08, n_samples // 2)
enmo = np.concatenate([enmo_sleep, enmo_wake])

labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))  # 0=sleep, 1=wake

df = pd.DataFrame({
    'anglez': anglez,
    'enmo': enmo,
    'label': labels
})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("=" * 50)
print("SLEEP DETECTION - ML PIPELINE")
print("=" * 50)

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n[1] Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nClass Distribution:\n{df['label'].value_counts().rename({0: 'Sleep', 1: 'Wake'})}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Stats:\n{df.describe()}")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Distribution of anglez
axes[0].hist(df[df['label'] == 0]['anglez'], bins=30, alpha=0.7, label='Sleep', color='blue')
axes[0].hist(df[df['label'] == 1]['anglez'], bins=30, alpha=0.7, label='Wake', color='orange')
axes[0].set_title('Anglez Distribution')
axes[0].set_xlabel('Anglez')
axes[0].legend()

# Distribution of enmo
axes[1].hist(df[df['label'] == 0]['enmo'], bins=30, alpha=0.7, label='Sleep', color='blue')
axes[1].hist(df[df['label'] == 1]['enmo'], bins=30, alpha=0.7, label='Wake', color='orange')
axes[1].set_title('ENMO Distribution')
axes[1].set_xlabel('ENMO')
axes[1].legend()

# Scatter plot
axes[2].scatter(df[df['label'] == 0]['anglez'], df[df['label'] == 0]['enmo'],
                alpha=0.3, label='Sleep', color='blue', s=10)
axes[2].scatter(df[df['label'] == 1]['anglez'], df[df['label'] == 1]['enmo'],
                alpha=0.3, label='Wake', color='orange', s=10)
axes[2].set_title('Anglez vs ENMO')
axes[2].set_xlabel('Anglez')
axes[2].set_ylabel('ENMO')
axes[2].legend()

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=120)
plt.show()
print("\n[EDA plots saved as eda_plots.png]")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[2] Feature Engineering...")
df['anglez_abs'] = df['anglez'].abs()
df['enmo_log']   = np.log1p(df['enmo'])
df['movement']   = df['anglez_abs'] * df['enmo']

X = df.drop('label', axis=1)
y = df['label']

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 5. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────
print("\n[3] Training Models...")

models = {
    'Random Forest':  RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN':            KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes':    GaussianNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    results[name] = {
        'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall':    round(recall_score(y_test, y_pred) * 100, 2),
        'F1-Score':  round(f1_score(y_test, y_pred) * 100, 2),
    }

# ─────────────────────────────────────────────
# 6. RESULTS
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL COMPARISON RESULTS")
print("=" * 50)
results_df = pd.DataFrame(results).T
print(results_df)

best_model_name = results_df['Accuracy'].idxmax()
print(f"\n✅ Best Model: {best_model_name} ({results_df.loc[best_model_name, 'Accuracy']}% accuracy)")

# Confusion Matrix for best model
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_sc)

print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, y_pred_best, target_names=['Sleep', 'Wake']))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sleep', 'Wake'],
            yticklabels=['Sleep', 'Wake'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=120)
plt.show()
print("[Confusion matrix saved as confusion_matrix.png]")
