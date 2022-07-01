from pygments import highlight
import streamlit as st
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import warnings
import numpy as np

warnings.filterwarnings('ignore')
# fill the missing values for numerical terms - mean
df = pd.read_csv("Loan Prediction Dataset.csv")

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

# fill the missing values for categorical terms - mode
df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
df['Married'] = df["Married"].fillna(df['Married'].mode()[0])
df['Dependents'] = df["Dependents"].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df["Self_Employed"].fillna(df['Self_Employed'].mode()[0])



df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome']+1)
df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome']+1)
df['LoanAmountLog'] = np.log(df['LoanAmount']+1)
df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term']+1)
df['Total_Income_Log'] = np.log(df['Total_Income']+1)





# drop unnecessary columns
cols = ['ApplicantIncome', 'CoapplicantIncome', "LoanAmount", "Loan_Amount_Term", "Total_Income", 'Loan_ID', 'CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)

from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status","Dependents"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# specify input and output attributes
X = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

modell = pickle.load(open('vector.pkl','rb'))


y_pred = modell.predict(x_test)


st.title("Home loan Prediction App")

Gender = st.number_input(  "Enter the Gender '0' for FEMALE '1' for MALE",0,1,0,1  )
Married = st.number_input("Enter the Marital Status '0' for NO '1' for YES",0,1,0,1)
Dependents= st.number_input("Enter the Number of Dependents 0-1-2-3",0,3,0,1)
Education = st.number_input("Enter the Education status '0' for Graduate '1' for Non-Graduate",0,1,0,1)
Self_Employed = st.number_input("Enter the Self Employed status '0' for NO '1' for YES",0,1,0,1)
Credit_History = st.number_input("Enter the Credit History '0' for BAD '1' for GOOD",0,1,0,1)
Property_Area= st.number_input("Enter the Property Area '0' for RURAL '1' for SEMI-URBAN '2' for URBAN",0,2,0,1)
ApplicantIncomeLog = st.number_input("Enter the Applicant Income")
LoanAmountLog = st.number_input("Enter the Loan Amount")
Loan_Amount_Term_Log = st.number_input("Enter the Loan Amount Term")


data = {'Name':[Gender],
        'Married':[Married],
        'Dependents':[Dependents],
        'Education':[Education],
        'Self_Employed':[Self_Employed],
        'Credit_History':[Credit_History],
        'Property_Area':[Property_Area],
        'ApplicantIncomeLog':[ApplicantIncomeLog],
        'LoanAmountLog':[LoanAmountLog],
        'Loan_Amount_Term_Log':[Loan_Amount_Term_Log],
        'Total_Income_Log':[ApplicantIncomeLog]}
        

uip = pd.DataFrame(data)
if st.button('Predict'):

    result = modell.predict(uip)[0]
    # 4. Display
    if result == 1:
        st.success("The loan is approved")
    else:
        st.error("The loan is rejected")

      

