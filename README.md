# Loan-Eligibility-Prediction
This project is designed to predict the eligibility of loan applicants based on various factors such as income, credit history, and marital status. By analyzing historical loan application data, the model helps to determine whether a loan application should be approved or not.


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data_Loading](#data_loading)
- [Data_Cleaning_and_Preprocessing](#data_cleaning_and_preprocessing)
- [Normalization](#normalization)
- [Modeling](#modeling)
- [Results](#results)
- [License](#license)

## introduction
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
To run this project, you need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
  
You can install these libraries using pip:

     pip install pandas numpy scikit-learn seaborn matplotlib
  
OR 
Clone the repository and install the required libraries:

     git clone https://github.com/chandkund/Loan-Eligibility-Prediction.git

## Data_Loading
First, load the dataset using pandas:
```python
raw_data = pd.read_csv("D:\\Data_Science_Project\\Project_5\\train.csv")
raw_data.head()
df = raw_data.copy()
df.info()  # Check the dataset information
```

## Data_Cleaning_and_Preprocessing
The following steps are performed for data cleaning and preprocessing:

-  Handle missing values in numerical and categorical columns using SimpleImputer.
-  Visualize the cleaned data using box plots and histograms.
-  Normalize and standardize the data for better model performance.
```python 
sns.pairplot(df)
plt.show()
pd.crosstab(df['Credit_History'],df['Loan_Status'],margins = True)
df.boxplot(column = 'ApplicantIncome')
df['ApplicantIncome'].head()
df['ApplicantIncome'].hist(bins= 20)
plt.show()
df.boxplot(column = 'LoanAmount')
plt.show()
df['LoanAmount'].hist(bins = 20)
plt.show()
# Normalization
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins = 20)
plt.show()
df.isnull().sum()
```
- **Imputing missing values***:
```python 
df["Gender"].fillna(df["Gender"].mode()[0],inplace =True)
df["Married"].fillna(df["Married"].mode()[0],inplace =True)
df["Dependents"].fillna(df["Dependents"].mode()[0],inplace =True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0],inplace =True)
df["LoanAmount"].fillna(df["LoanAmount"].mean(),inplace =True)
df["LoanAmount_log"].fillna(df["LoanAmount_log"].mean(),inplace =True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0],inplace =True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0],inplace =True)
df.isnull().sum()
# Heatmap => All numerical data 
num_df = df.select_dtypes(include=['number']) 
sns.heatmap(num_df.corr(), annot=True)
plt.title("Correlation Heatmap for all Numerial Variables")
```

 ## Normalization
```python 
df["TotalIncome"] = df['ApplicantIncome'] +df["CoapplicantIncome"]
df["TotalIncome_log"] =np.log(df["TotalIncome"])
df["TotalIncome_log"].hist(bins = 20)
plt.show()
```

 - **Features and target**:
 ```python
X = df.iloc[:,np.r_[1:5,9:11,13:15]].values
Y = df.iloc[:,12].values
```

- **Split the data**:
```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state= 0)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
```
- **LabelEncoder**:

```python
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(0,5):
    X_train[:,i] =labelencoder_X.fit_transform(X_train[:,i])
X_train[:,7] =labelencoder_X.fit_transform(X_train[:,7])
X_train[:5]
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
Y_train =labelencoder_y.fit_transform(Y_train)
Y_train[:5]
for i in range(0,5):
    X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i])

X_test[:5]

X_test[:,7] =labelencoder_X.fit_transform(X_test[:,7])
Y_test =labelencoder_y.fit_transform(Y_test)
Y_test[:5]
```

- **Standardizatio of Data**:
```python
from sklearn.preprocessing import StandardScaler 

scaled = StandardScaler()
X_train = scaled.fit_transform(X_train)
X_text = scaled.fit_transform(X_test)
X_train
```

Different regression models are built and evaluated:

- LogisticRegression
- Support Vector Machine
- RDecisionTreeClassifier
- EKNeighborsClassifier

## Modeling

- **Model_1:Logistic Regression**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

model_1 = LogisticRegression()
model_1.fit(X_train,Y_train)
```
- **Model_1:Evaluation**: 
```python
pred1 = model_1.predict(X_test)
score1 = accuracy_score(pred,Y_test)
print(f'Accuracy: {score * 100:.2f}%')
```

- ***Model_2: Support Vector Machine**:
```python
from sklearn.svm import SVC
model_2 = SVC()
model_2.fit(X_train,Y_train)
```
- **Model_2:Evaluation**:
```python
pred2 = model_2.predict(X_test)
score2  = accuracy_score(pred2,Y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
```
- **Model 3 .DecisionTreeClassification**:

```python
from sklearn.tree import DecisionTreeClassifier

Model_3 = DecisionTreeClassifier()
Model_3.fit(X_train,Y_train)
```
- **Model_3:Evaluation**: 

```python
pred3 = Model_3.predict(X_test)
score3 = accuracy_score(pred3,Y_test)
print(f'Accuracy: {score3 * 100:.2f}%')
```
- **Model 4.KNeighborsClassifier**:

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
```
- **Model_4:Evaluation**: 
```python
pred4 = knn.predict(X_test)
score4 = accuracy_score(pred4,Y_test)
print(f'Accuracy: {score4 * 100:.2f}%')
```


## Results
The models are evaluated based on Mean Squared Error (MSE). Below are the MSE results for each model:

- LogisticRegression accuracy_score  : 82.93%
- Support Vector Machine  accuracy_score  :82.93%
- DecisionTreeClassifier accuracy_score  :73.17%
- KNeighborsClassifier accuracy_score  : 79.67%

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


