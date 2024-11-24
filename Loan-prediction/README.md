# Loan Approval Prediction Using Ensemble Learning

## Project Overview

This project aims to predict whether a bank should approve a loan application based on the applicant's financial profile using ensemble learning techniques. The dataset used for the analysis includes attributes such as FICO scores, annual income, debt-to-income ratio, credit policies, and loan purpose. The models are evaluated based on their accuracy, recall, and precision, with Random Forest emerging as the best-performing model.

---

## Dataset Details

- **Source**: `loan_data.csv`
- **Number of Records**: 9,578
- **Number of Attributes**: 14

### Key Attributes:
1. **credit.policy**: Whether the applicant meets the bank's credit underwriting policy.
2. **purpose**: Purpose of the loan (e.g., debt consolidation, credit card, etc.).
3. **int.rate**: Interest rate of the loan.
4. **installment**: Monthly installment.
5. **log.annual.inc**: Logarithmic annual income.
6. **dti**: Debt-to-income ratio.
7. **fico**: FICO credit score.
8. **days.with.cr.line**: Length of credit history in days.
9. **revol.bal**: Revolving balance.
10. **revol.util**: Percentage of available credit utilized.
11. **inq.last.6mths**: Number of credit inquiries in the last 6 months.
12. **delinq.2yrs**: Number of times the applicant was 30+ days past due in the last 2 years.
13. **pub.rec**: Number of derogatory public records.
14. **not.fully.paid**: Target variable (1 if the loan was not fully paid, 0 otherwise).

---

## Data Preprocessing

1. **Handling Null Values**: Verified there are no missing values.
2. **Encoding Categorical Features**: 
   - The `purpose` column was label-encoded for numerical processing.
3. **Exploratory Data Analysis (EDA)**: 
   - Histograms and count plots were used to visualize relationships between features and the target variable.
   - Correlation heatmap identified `int.rate`, `fico`, and `inq.last.6mths` as highly correlated with the target variable.

---

## Model Development and Evaluation

### Models Used:
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

#### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score

#### Train-Test Split:
- **Training Set**: 70%
- **Test Set**: 30%

### Results Summary:

| Model                  | Test Accuracy | Observations                                                   |
|------------------------|---------------|----------------------------------------------------------------|
| Decision Tree          | 84.58%        | Moderate accuracy but unable to classify minority class.       |
| Random Forest          | **84.60%**    | Best accuracy with improved performance for minority class.    |
| Gradient Boosting      | 84.40%        | Slightly lower accuracy, limited impact on minority class.     |

**Best Model**: **Random Forest Classifier**

- Random Forest achieved the best accuracy of **84.6%**.
- It provides better generalization by leveraging multiple decision trees.

---

## Key Insights

1. **FICO Scores and Credit Policies** are critical factors in predicting loan repayment capability.
2. The dataset is imbalanced, with significantly fewer instances of `not.fully.paid=1`. This impacts recall for the minority class.
3. Ensemble methods (Random Forest) outperform single-tree models, leveraging bagging for improved robustness.

---

## Project Structure

```plaintext
├── data/
│   └── loan_data.csv    # Dataset used for analysis
├── notebooks/
│   └── loan_analysis.ipynb   # Jupyter notebook with code and analysis
├── outputs/
│   ├── plots/          # Visualizations generated during EDA
│   └── results.txt     # Model evaluation results
└── README.md           # Project overview (this file)
```

---

## How to Run the Project

1. **Setup**:
   - Install Python 3.x and the required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
   - Install dependencies via `pip install -r requirements.txt`.

2. **Data**:
   - Ensure the `loan_data.csv` file is placed in the `data/` directory.

3. **Execution**:
   - Open `loan_analysis.ipynb` in Jupyter Notebook.
   - Run all cells sequentially to reproduce the analysis.

---

## Future Improvements

1. Address data imbalance using techniques like SMOTE or undersampling.
2. Explore additional algorithms, such as XGBoost or LightGBM, for enhanced performance.
3. Fine-tune hyperparameters of Random Forest and Gradient Boosting models for better recall.

---
