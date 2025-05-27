# Housing Price Prediction using Machine Learning

This project aims to predict house prices using various machine learning regression models. It uses a dataset (`Housing.csv`) that includes both numerical and categorical features about housing data. The models used include:

- Random Forest Regressor
- Linear Regression
- Gradient Boosting Regressor (with GridSearchCV tuning)


## Project Structure

- **data/Housing.csv**: Dataset containing housing features and target price.
- **main.py** (or similar script): Python script containing data preprocessing, model training, evaluation, and hyperparameter tuning.


## Workflow Summary

1. **Load the Data**
   - Read the dataset into a pandas DataFrame.
   - Display basic info and preview the first few rows.

2. **Split Features and Target**
   - `X` = Features (all columns except "price")
   - `y` = Target variable ("price")

3. **Preprocessing Pipelines**
   - Numerical features are imputed with median and scaled.
   - Categorical features are imputed with most frequent value and one-hot encoded.
   - Both pipelines are combined using `ColumnTransformer`.

4. **Train-Test Split**
   - Data is split into 80% training and 20% testing sets.

5. **Model Pipelines**
   - Create pipelines for three regression models:
     - Random Forest
     - Linear Regression
     - Gradient Boosting
   - Each pipeline includes preprocessing + the model.

6. **Model Evaluation**
   - Each model is trained and tested.
   - Evaluation metrics:
     - MAE (Mean Absolute Error)
     - MSE (Mean Squared Error)
     - RMSE (Root Mean Squared Error)
     - R² Score (Coefficient of Determination)

7. **Hyperparameter Tuning (GridSearchCV)**
   - Applied to `GradientBoostingRegressor` using `GridSearchCV`.
   - Parameters tuned:
     - `n_estimators` (100, 200, 300)
     - `learning_rate` (0.05, 0.1, 0.2)
     - `max_depth` (3, 4, 5)
   - Best model is selected based on lowest RMSE.

8. **Results Visualization**
   - A heatmap is plotted to visualize performance of different hyperparameter combinations using seaborn.


## Evaluation Metrics

- **MAE**: Measures average absolute errors.
- **MSE**: Measures average squared errors.
- **RMSE**: Root of MSE for interpretability in original units.
- **R² Score**: Indicates how well the model explains variance in target.

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
