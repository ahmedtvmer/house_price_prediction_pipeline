# 🏡 House Price Prediction: Hybrid Ensemble Architecture

## 🎯 Project Goal
To develop a production-ready Machine Learning pipeline that beats standard baselines by combining the non-linear power of **Gradient Boosting** with the stability of **Regularized Linear Regression**. The system is engineered to handle "small data" constraints (1,460 rows) while maximizing generalization through aggressive feature engineering and weighted model ensembling.

## 📊 Performance Summary (Production Model)

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Architecture** | **Hybrid Ensemble** | 70% CatBoost + 30% Lasso Regression |
| **RMSE (Log Scale)** | **0.1275** | Competitive performance, effectively minimizing variance by blending distinct model errors. |
| **$R^2$ Score** | **0.9129** | Explains >91% of the variance in sale prices on unseen test data. |

---

## 🛠️ Engineering & Architecture

### 1. The "Hybrid" Pipeline Strategy
Instead of relying on a single algorithm, we implemented a **Voting Regressor** logic (Weighted Average) to capture different aspects of the data:
* **CatBoost (70% Weight):** Captures complex, non-linear relationships and interactions (e.g., specific neighborhoods penalizing price only if the condition is poor). Uses **Native Categorical Handling** (no One-Hot Encoding) to reduce dimensionality.
* **Lasso Regression (30% Weight):** Captures strong linear baselines (e.g., "More Area = More Money") and handles feature selection via L1 regularization. Uses standard **One-Hot Encoding** and **Scaling**.

### 2. Feature Engineering
Features were generated in two stages to prevent data leakage while maximizing signal:
* **Pandas (Pre-Split):** "Safe" mathematical combinations.
    * `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`.
    * `TotalBath` = Weighted sum of Full/Half baths.
* **Pipeline (Post-Split):**
    * **Custom `AgeFeatureCreator`:** Dynamically transforms `YearBuilt` and `YrSold` into `HouseAge` and `GarageAge`.

### 3. Key Technologies
* **Algorithms:** `CatBoostRegressor`, `sklearn.linear_model.Lasso`
* **Preprocessing:** `ColumnTransformer`, `SimpleImputer` (Median/Constant), `StandardScaler` (Lasso only).
* **Target Transformation:** Log-transform ($\ln(1 + y)$) applied to `SalePrice` to correct right-skewness.

### 4. Hyperparameter Tuning
* **CatBoost:** Tuned for stability on small data.
    * `learning_rate`: **0.04** (Slow learning to prevent overfitting).
    * `depth`: **4** (Shallow trees to force generalization).
    * `l2_leaf_reg`: **3** (Regularization).
* **Lasso:** Tuned `alpha` (**0.0005**) to balance penalty vs. fit.

### 5. Deployment Structure
The final model is saved as a dictionary package containing both pipelines and their respective weights, ensuring the ensemble logic is preserved during inference:

```python
full_model_package = {
    'catboost_pipe': pipeline,
    'lasso_pipe': lasso_pipeline,
    'catboost_weight': 0.7,
    'lasso_weight': 0.3
}
