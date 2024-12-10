import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("Telco_collected_dataset(FAR).csv")
df.drop(columns=['Churn1'], inplace=True)
df.drop(columns=['Churn'], inplace=True)
x=df.drop("Churn2",axis=1)
y=df["Churn2"]
tenure_mapping = {'Less than 1 Year': 0,'1 Year': 1,'2 Years': 2,'More than 2 Years': 3}
x['Tenure'] = x['Tenure'].replace(tenure_mapping)
categorical_cols = ['Gender','ServiceProvider', 'DataPlan', 'StreamTV', 'StreamMovies', 'TypeofPlan', 'PaymentMethod']
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_cols)], remainder='passthrough')
x = transformer.fit_transform(x)
x=pd.DataFrame(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.head(1))
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

st.set_page_config(page_title="Telco Churn Prediction")
st.title("Telco Churn Predictor")
gender=["Male","Female"]
tenure=["1 Year","2 Years","More than 2 Years","Less than 1 Year"]
SP=["Airtel","Jio","Vi (Vodafone Idea)","BSNL"]
DP=["4G","5G"]
TV=["Yes","No"]
TOP=["Pre-Paid","Post-Paid"]
PM=["UPI","Internet Banking","Credit Card","Debit Card"]
col1, col2, col3 = st.columns(3)

with col1:
  Gender=st.selectbox("Gender",options=gender)
  Age=st.number_input("Age")
  Tenure=st.selectbox("Tenure",options=tenure)

with col2:
  ServiceProvider=st.selectbox("Service Provider",options=SP)
  DataPlan=st.selectbox("Data Plan",options=DP)
  StreamTV=st.selectbox("Do you stream TV",options=TV)
  StreamMovies=st.selectbox("Do you stream Movies",options=TV)

with col3:
  TypeofPlan=st.selectbox("Type of Plan",options=TOP)
  PaymentMethod=st.selectbox("Payment Method",options=PM)
  MonthlyCharges=st.number_input("Enter your monthly charges")

pdf=pd.read_csv("Telco_collected_dataset2.csv")
pdf.drop(columns=['Churn1'], inplace=True)
pdf.drop(columns=['Churn'], inplace=True)
x=pdf.drop("Churn2",axis=1)
new={'Gender':Gender,'Age':Age,'Tenure':Tenure,'ServiceProvider':ServiceProvider,'DataPlan':DataPlan,'StreamTV':StreamTV,'StreamMovies':StreamMovies,'TypeofPlan':TypeofPlan,'PaymentMethod':PaymentMethod,'MonthlyCharges':MonthlyCharges}
x.loc[len(x)] = new
print(x.tail(1))
tenure_mapping = {'Less than 1 Year': 0,'1 Year': 1,'2 Years': 2,'More than 2 Years': 3}
x['Tenure'] = x['Tenure'].replace(tenure_mapping)
categorical_cols = ['Gender','ServiceProvider', 'DataPlan', 'StreamTV', 'StreamMovies', 'TypeofPlan', 'PaymentMethod']
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_cols)], remainder='passthrough')
x_trans = transformer.fit_transform(x)
x=pd.DataFrame(x_trans)
test=x.tail(1)
print(test)
t=clf.predict(test)
t=t[0]
t=str(t)
if(t=="1"):
  t="Yes"
else:
  t="No"
res=st.button("Predict")
if res:
 st.subheader("Churn")
 st.write(t)
