import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

# Data Upload Module
def upload_data():
    st.title("Enhanced College Attendance + Class Performance Predictor")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())
        return data
    return None

# Data Preprocessing Module
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

# Machine Learning Module
def train_model(data):
    st.subheader("Model Training")
    st.write("Splitting data into training and testing sets...")
    X = data.drop(columns=["final_grade", "risk_status"], errors="ignore")
    y = data["risk_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write("Training Random Forest Classifier...")
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    
    st.write("Model Evaluation:")
    st.text(classification_report(y_test, y_pred_rf))
    st.write("Accuracy:", accuracy_score(y_test, y_pred_rf))
    
    return model_rf

# Clustering Module (K-Means for Study Behavior)
def clustering_analysis(data):
    st.subheader("Clustering Students by Study Behavior")
    if st.checkbox("Perform Clustering Analysis"):
        kmeans = KMeans(n_clusters=3, random_state=42)
        features = data.drop(columns=["risk_status", "final_grade"], errors="ignore")
        data["Cluster"] = kmeans.fit_predict(features)
        st.write("Clustering Completed. Clustered Data Preview:")
        st.dataframe(data)
        
        plt.title("Cluster Visualization (First Two Features)")
        plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=data["Cluster"])
        plt.xlabel(features.columns[0])
        plt.ylabel(features.columns[1])
        st.pyplot(plt.gcf())

# Prediction Dashboard Module
def prediction_dashboard(model, data):
    st.subheader("Prediction Dashboard")
    st.write("Upload new student data for predictions:")
    prediction_file = st.file_uploader("Upload CSV file for predictions", type=["csv", "xlsx"])
    if prediction_file:
        if prediction_file.name.endswith("csv"):
            new_data = pd.read_csv(prediction_file)
        else:
            new_data = pd.read_excel(prediction_file)
        
        st.write("New Data Preview:")
        st.dataframe(new_data.head())
        
        X_new = new_data
        predictions = model.predict(X_new)
        new_data["Predicted Risk Status"] = predictions
        st.write("Predictions:")
        st.dataframe(new_data)
        
        st.write("Download Predictions:")
        st.download_button("Download CSV", new_data.to_csv(index=False), file_name="predictions.csv")

# Visualization Module
def visualization_module(data):
    st.subheader("Visual Insights")
    if "attendance" in data.columns and "final_grade" in data.columns:
        st.write("Attendance vs Final Grade")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data["attendance"], y=data["final_grade"], hue=data["risk_status"])
        plt.xlabel("Attendance")
        plt.ylabel("Final Grade")
        plt.title("Attendance vs Final Grade")
        st.pyplot(plt.gcf())
    
    if st.checkbox("Show Correlation Heatmap"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        st.pyplot(plt.gcf())

# Personalized Recommendations Module
def personalized_recommendations(data, model):
    st.subheader("Personalized Recommendations")
    st.write("Feature Importance Analysis:")
    X = data.drop(columns=["final_grade", "risk_status"], errors="ignore")
    y = data["risk_status"]
    importance = permutation_importance(model, X, y, scoring="accuracy")
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance.importances_mean
    }).sort_values(by="Importance", ascending=False)
    st.dataframe(feature_importance)
    
    st.write("Top Recommendations:")
    if "attendance" in data.columns:
        low_attendance_students = data[data["attendance"] < 0.5]
        for _, row in low_attendance_students.iterrows():
            st.write(f"Student ID {row['student_id']}: Increase attendance by {0.5 - row['attendance']:.2%} to reach safe zone.")

# Report Generator Module
def generate_report(data):
    st.subheader("Generate Report")
    st.write("Student-Level Summaries")
    student_summary = data.groupby("student_id").mean()
    st.dataframe(student_summary)
    
    st.write("Export Report to PDF")
    if st.button("Generate PDF"):
        st.write("PDF report generation feature is under development.")

# Main Application
def main():
    data = upload_data()
    if data is not None:
        data = preprocess_data(data)
        model = train_model(data)
        clustering_analysis(data)
        prediction_dashboard(model, data)
        visualization_module(data)
        personalized_recommendations(data, model)
        generate_report(data)

if __name__ == "__main__":
    main()
