import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from preprocess import create_preprocessor


# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv("data/customer_churn.csv")

df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# 2. Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 3. Preprocessing + Model
# =========================

preprocessor = create_preprocessor(X_train)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)

# Train
grid_search.fit(X_train, y_train)

# =========================
# 4. Evaluate Model
# =========================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 5. Save Model
# =========================

joblib.dump(best_model, "models/model.pkl")

print("\n✅ Model trained and saved successfully!")
print("Best Parameters:", grid_search.best_params_)
