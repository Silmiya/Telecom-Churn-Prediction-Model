import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Generate dynamic remarks based on the prediction
def generate_remark(prediction, user_data, telco_data):
    if prediction == 1:  # Churn
        if user_data['Tenure'] in ['Less than 1 Year', '1 Year']:
            tenure_remark = "The short tenure indicates potential dissatisfaction or early-stage disengagement."
        else:
            tenure_remark = "Despite a longer tenure, there might be specific concerns leading to churn."

        high_charges_remark = "High monthly charges could be a factor. Consider providing cost-effective solutions." \
            if user_data['MonthlyCharges'] > telco_data['MonthlyCharges'].mean() else ""

        return f"""
        ### Remark
        The customer is likely to churn. {tenure_remark} {high_charges_remark}
        Consider addressing concerns related to service quality, pricing, or engagement to retain them.
        """
    else:  # No Churn
        loyal_remark = "The customer's longer tenure suggests satisfaction with services." \
            if user_data['Tenure'] in ['More than 2 Years', '2 Years'] else ""

        optimal_plan_remark = "Their current plan and charges seem to meet their needs effectively." \
            if user_data['MonthlyCharges'] <= telco_data['MonthlyCharges'].mean() else ""

        return f"""
        ### Remark
        The customer is unlikely to churn. {loyal_remark} {optimal_plan_remark}
        Maintain engagement through loyalty rewards and consistent service quality.
        """

# Initialize session state for navigation
if "show_remark" not in st.session_state:
    st.session_state.show_remark = False
    st.session_state.prediction = None
    st.session_state.user_data = None

# Load the dataset
df = pd.read_csv("Telco_collected_dataset.csv")

# Drop unnecessary columns
df.drop(columns=['Churn1', 'Churn'], inplace=True)

# Define the target variable and features
X = df.drop("Churn2", axis=1)
y = df["Churn2"]

# Fix 'Tenure' column with mapping
tenure_mapping = {
    'Less than 1 Year': 0,
    '1 Year': 1,
    '2 Years': 2,
    'More than 2 Years': 3,
    '3 Years': 3,  # Map '3 Years' to 'More than 2 Years'
    '2 Year': 2    # Correct '2 Year' to '2 Years'
}
X['Tenure'] = X['Tenure'].map(tenure_mapping)

# Drop rows with unmapped Tenure values
X.dropna(subset=['Tenure'], inplace=True)

# Define categorical columns for encoding
categorical_cols = ['Gender', 'ServiceProvider', 'DataPlan', 'StreamTV', 'StreamMovies', 'TypeofPlan', 'PaymentMethod']

# Apply OneHotEncoder to categorical columns
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_cols)], remainder='passthrough')
X_transformed = transformer.fit_transform(X)

# Convert the transformed data back to DataFrame for better handling
X_transformed = pd.DataFrame(X_transformed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Main Page (Input and Prediction)
if not st.session_state.show_remark:
    st.set_page_config(page_title="Telco Churn Prediction")
    st.title("Telco Churn Predictor")

    # Input fields for user data
    gender = ["Male", "Female"]
    tenure = ["Less than 1 Year", "1 Year", "2 Years", "More than 2 Years"]
    SP = ["Airtel", "Jio", "Vi (Vodafone Idea)", "BSNL"]
    DP = ["4G", "5G"]
    TV = ["Yes", "No"]
    TOP = ["Pre-Paid", "Post-Paid"]
    PM = ["UPI", "Internet Banking", "Credit Card", "Debit Card"]

    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox("Gender", options=gender)
        Age = st.number_input("Age")
        Tenure = st.selectbox("Tenure", options=tenure)

    with col2:
        ServiceProvider = st.selectbox("Service Provider", options=SP)
        DataPlan = st.selectbox("Data Plan", options=DP)
        StreamTV = st.selectbox("Do you stream TV", options=TV)
        StreamMovies = st.selectbox("Do you stream Movies", options=TV)

    with col3:
        TypeofPlan = st.selectbox("Type of Plan", options=TOP)
        PaymentMethod = st.selectbox("Payment Method", options=PM)
        MonthlyCharges = st.number_input("Enter your monthly charges")

    # Prepare user input
    new_input = {
        'Gender': Gender, 'Age': Age, 'Tenure': Tenure, 'ServiceProvider': ServiceProvider,
        'DataPlan': DataPlan, 'StreamTV': StreamTV, 'StreamMovies': StreamMovies,
        'TypeofPlan': TypeofPlan, 'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges
    }

    # Add new input to DataFrame for transformation
    user_df = X.iloc[:0].copy()
    user_df.loc[0] = new_input
    user_df['Tenure'] = user_df['Tenure'].map(tenure_mapping)
    user_transformed = transformer.transform(user_df)

    # Predict churn
    res = st.button("Predict")
    if res:
        prediction = clf.predict(user_transformed)[0]
        st.session_state.prediction = prediction
        st.session_state.user_data = new_input
        st.session_state.show_remark = True

# Remark Page (Results and Remarks)
else:
    st.title("Prediction Result")
    prediction = st.session_state.prediction
    user_data = st.session_state.user_data

    # Generate dynamic remark
    remark = generate_remark(prediction, user_data, df)

    # Display prediction result
    churn_status = "Yes" if prediction == 1 else "No"
    st.subheader("Churn Prediction")
    st.write(churn_status)

    # Display generated remark
    st.markdown(remark)

    # Button to go back to the main page
    if st.button("Go Back"):
        st.session_state.show_remark = False
