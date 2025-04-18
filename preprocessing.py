import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st

def upload_data():
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())
        return data
    return None

def preprocess_data(data):
    st.subheader("Data Preprocessing")
    st.write("Handling missing values...")
    data.fillna(data.mean(), inplace=True)
    
    st.write("Scaling numerical features...")
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    st.write("Encoding categorical variables...")
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    st.write("Preprocessed Data Preview:")
    st.dataframe(data.head())
    return data
