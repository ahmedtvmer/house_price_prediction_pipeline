# 🏡 House Price Prediction Pipeline

## 🎯 Project Goal
To develop a robust, end-to-end Machine Learning pipeline using **Scikit-Learn** for predicting residential house prices. The primary focus was on establishing a scalable, production-ready preprocessing workflow while addressing common data science challenges like **data leakage** and **target skewness**.

## 📊 Performance Summary (Baseline Model)
| Metric | Value | Note |
| :--- | :--- | :--- |
| **Model** | Linear Regression | V1 Baseline |
| **$R^2$ Score** | **0.9153** | Explains over 91% of the variance in house prices. |
| **RMSE (Log Scale)** | **0.1257** | An excellent score, placing the model within the expected competitive range for this baseline. |

---

## 🛠️ Key Technologies & Concepts

* **Python:** 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Joblib
* **Target Transformation:** Logarithm ($\ln(1 + y)$) used on `SalePrice` to normalize distribution and stabilize variance, a mandatory step for Linear Regression.
* **Pipeline Architecture:** Implemented a single `sklearn.pipeline.Pipeline` to encapsulate all preprocessing and modeling steps, ensuring zero data leakage.
* **Custom Transformers:** Created the `AgeFeatureCreator` custom class to perform **Feature Engineering** (Year $\to$ Age) directly within the pipeline structure.
* **Imputation Strategy:** Imputed numerical data with the **median** and categorical data with `"None"` (to preserve the informative nature of missing values).
* **Feature Encoding:** Applied **Ordinal Encoding** (manual mapping) to rank-based quality features (`KitchenQual`, `ExterQual`) and **One-Hot Encoding** to nominal features (`Neighborhood`).
