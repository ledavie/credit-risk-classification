# credit-risk-classification
Module Twenty Challenge
# Credit Risk Classification with Logistic Regression

## Overview of the Analysis

The purpose of this analysis was to build and evaluate a logistic regression model to predict the likelihood of a loan becoming high-risk. The data included financial features such as income, debt-to-income ratio, and loan amount, with the target variable being `loan_status`. A value of `0` indicates a healthy loan, while a value of `1` signifies a high-risk loan.

We followed the standard machine learning workflow:

* Loaded and reviewed the dataset (`lending_data.csv`)
* Split the data into features (`X`) and labels (`y`)
* Split the dataset into training and testing subsets
* Trained a logistic regression model on the training data
* Evaluated the model’s performance on the test data

The primary algorithm used in this analysis was `LogisticRegression` from `scikit-learn`.

## Results

* **Accuracy Score**: High (model tends to predict the majority class well)
* **Precision Score**:
  * For class `0` (healthy loan): Very high
  * For class `1` (high-risk loan): Very low
* **Recall Score**:
  * For class `0`: Very high
  * For class `1`: Very low

This means:
- The model is excellent at identifying loans that are not risky.
- The model performs poorly at identifying loans that are high-risk.

## Summary

The logistic regression model showed strong performance for predicting healthy loans but struggled significantly to identify high-risk loans. This is likely due to a **class imbalance**, where the number of `0`s (healthy loans) far exceeds the number of `1`s (high-risk loans).

**Recommendation:**

We do **not recommend** this model for production use in its current form because it fails to identify high-risk loans — the very predictions that are most valuable for risk management.

**Next steps for improvement might include:**

* Resampling techniques (e.g., SMOTE) to address class imbalance
* Trying alternative models such as Random Forest or Gradient Boosting
* Hyperparameter tuning and feature engineering to improve model performance
