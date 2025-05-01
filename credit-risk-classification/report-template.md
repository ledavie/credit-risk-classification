# Module 12 Report Template

## Overview of the Analysis

The purpose of this analysis was to assess the risk level of loan applications using machine learning. Specifically, the goal was to predict whether a loan is **healthy (0)** or **high-risk (1)** based on various financial features.

The dataset included borrower information such as:
- Monthly debt
- Monthly income
- Loan amount
- Debt-to-income ratio
- Number of accounts

The target variable, `loan_status`, was a binary classification where:
- `0` = Healthy loan
- `1` = High-risk loan

The machine learning process followed these main steps:
- Load and inspect the data
- Separate the features (`X`) and target labels (`y`)
- Split the data into training and testing sets
- Train a logistic regression model on the training data
- Evaluate model performance using accuracy, precision, recall, and a confusion matrix

We used the `LogisticRegression` algorithm from `scikit-learn` for this task.



## Results

* **Machine Learning Model 1: Logistic Regression**
  * **Accuracy Score**: 
    - The overall correctness of predictions on the test data.
  * **Precision Score**:
    - For class `0` (healthy loans): High
    - For class `1` (high-risk loans): Low
  * **Recall Score**:
    - For class `0`: High
    - For class `1`: Low


## Summary

The logistic regression model demonstrated strong performance in identifying healthy loans but was weak in detecting high-risk loans. This is evident from the low precision and recall scores for the `1` class.

Since identifying high-risk loans is critical in managing credit risk, the model’s inability to do so reliably limits its effectiveness. The likely cause is **class imbalance**, with significantly more healthy loans than risky ones in the dataset.

## Recommendation

- At this stage, the model is **not recommended** for production use.
- To improve performance, especially on high-risk loans, we recommend:
  - Addressing class imbalance (e.g., using oversampling/undersampling techniques)
  - Testing other algorithms such as Random Forest or Gradient Boosting
  - Applying hyperparameter tuning and scaling features appropriately