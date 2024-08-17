# Loan-Eligibility-Prediction
This project is designed to predict the eligibility of loan applicants based on various factors such as income, credit history, and marital status. By analyzing historical loan application data, the model helps to determine whether a loan application should be approved or not.
# Loan Eligibility Prediction

## Overview
This project is designed to predict the eligibility of loan applicants based on various factors such as income, credit history, and marital status. By analyzing historical loan application data, the model helps to determine whether a loan application should be approved or not.

## Dataset
The dataset used for this project includes the following columns:
- **Loan_ID**: Unique identifier for each loan application.
- **Gender**: Applicant's gender (Male/Female).
- **Married**: Marital status of the applicant (Yes/No).
- **Dependents**: Number of dependents (0, 1, 2, 3+).
- **Education**: Education level of the applicant (Graduate/Not Graduate).
- **Self_Employed**: Whether the applicant is self-employed (Yes/No).
- **ApplicantIncome**: Income of the applicant.
- **CoapplicantIncome**: Income of the co-applicant (if any).
- **LoanAmount**: Requested loan amount.
- **Loan_Amount_Term**: Term of the loan in months.
- **Credit_History**: Whether the applicant has a credit history (1: Yes, 0: No).
- **Property_Area**: Area of the property (Urban/Semiurban/Rural).
- **Loan_Status**: Loan approval status (Y: Yes, N: No).

## Installation
To run this project, you'll need to have Python and the following libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

