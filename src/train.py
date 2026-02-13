import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from preprocess import create_preprocessor


# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv("data/customer_churn.csv")

# Drop ID column
df = df.drop("customerID", axis=1)

# Fix TotalCharges (it contains blank strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df = df.dropna()

# =========================
# 2. Split Features & Target
# =========================

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# 3. Create Preprocessing
# =========================

preprocessor = create_preprocessor(X)

# =========================
# 4. Create Pipeline
# =========================

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# =========================
# 5. Hyperparameter Tuning
# =========================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)

# Train model
grid_search.fit(X, y)

# =========================
# 6. Save Model
# =========================

joblib.dump(grid_search.best_estimator_, "models/model.pkl")

print("✅ Model trained and saved successfully!")
print("Best Parameters:", grid_search.best_params_)
