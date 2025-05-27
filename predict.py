
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("data/Housing.csv")  
print(df.head())
print(df.info())

X = df.drop("price", axis=1)
y = df["price"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

pipelines = {
    name: Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", model)
    ]) for name, model in models.items()
}

results = []

for name, pipeline in pipelines.items():
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R^2": r2
    })

#Gradient boosting
results_df = pd.DataFrame(results) 
print(results_df.sort_values(by="RMSE"))

param_grid = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__learning_rate": [0.05, 0.1, 0.2],
    "regressor__max_depth": [3, 4, 5]
}

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42))
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Best Model RMSE: {rmse:.2f}")
print(f"Best Model R² Score: {r2:.4f}")

# Convert cv_results_ to DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Extract only relevant columns
pivot_table = results_df.pivot_table(
    index="param_regressor__n_estimators",
    columns="param_regressor__learning_rate",
    values="mean_test_score"
)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(-pivot_table, annot=True, fmt=".0f", cmap="viridis")
plt.title("GridSearchCV: RMSE for Gradient Boosting")
plt.xlabel("Learning Rate")
plt.ylabel("Number of Estimators")
plt.tight_layout()
plt.show()
