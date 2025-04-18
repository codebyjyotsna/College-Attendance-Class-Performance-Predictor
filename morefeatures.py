import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import shap

# Data Upload Module
def upload_data():
    st.title("Advanced College Attendance + Class Performance Predictor")
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
    
    return model_rf, X_train

# Behavioral Analytics
def behavioral_analytics(data):
    st.subheader("Behavioral Analytics")
    st.write("Analyzing Assignment Submission Frequency vs. Performance...")
    if "assignment_submissions" in data.columns and "final_grade" in data.columns:
        sns.boxplot(data=data, x="assignment_submissions", y="final_grade")
        plt.title("Assignment Submission Frequency vs. Final Grade")
        st.pyplot(plt.gcf())
    else:
        st.write("Required columns not found in the dataset.")

# Attendance Heatmap
def attendance_heatmap(data):
    st.subheader("Attendance Heatmap")
    if "date" in data.columns and "attendance" in data.columns:
        attendance_pivot = data.pivot_table(index="date", columns="student_id", values="attendance", fill_value=0)
        sns.heatmap(attendance_pivot, cmap="YlGnBu", cbar=True)
        plt.title("Attendance Heatmap")
        st.pyplot(plt.gcf())
    else:
        st.write("Required columns (date, attendance) not found in the dataset.")

# Explainable AI with SHAP
def explainable_ai(model, X_train):
    st.subheader("Explainable AI: Feature Importance using SHAP")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    st.write("Feature Importance Summary Plot:")
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    st.pyplot(plt.gcf())

# Leaderboards
def leaderboards(data):
    st.subheader("Leaderboards")
    if "final_grade" in data.columns:
        top_students = data.sort_values(by="final_grade", ascending=False).head(10)
        st.write("Top 10 Students by Final Grade:")
        st.dataframe(top_students[["student_id", "final_grade"]])
    if "attendance" in data.columns:
        top_attendance = data.sort_values(by="attendance", ascending=False).head(10)
        st.write("Top 10 Students by Attendance:")
        st.dataframe(top_attendance[["student_id", "attendance"]])

# Achievement Badges
def achievement_badges(data):
    st.subheader("Achievement Badges")
    if "attendance" in data.columns and "final_grade" in data.columns:
        for _, row in data.iterrows():
            if row["attendance"] >= 0.9:
                st.write(f"🎖️ Student {row['student_id']} achieved 'Perfect Attendance'!")
            if row["final_grade"] >= 0.9:
                st.write(f"🏆 Student {row['student_id']} achieved 'Top Scorer'!")

# Email Alerts
def send_email_alerts(data):
    st.subheader("Email Alerts for At-Risk Students")
    if st.checkbox("Send Email Alerts"):
        at_risk_students = data[data["risk_status"] == 1]  # Assuming 1 indicates "At-Risk"
        for _, student in at_risk_students.iterrows():
            send_email(student["student_id"], student["email"])
        st.write("Email alerts sent to all at-risk students!")

def send_email(student_id, email):
    sender_email = "your_email@example.com"
    sender_password = "your_password"
    subject = "Academic Performance Alert"
    body = f"Dear Student ID {student_id},\n\nYou have been flagged as at-risk. Please take necessary actions to improve your performance.\n\nBest Regards,\nAcademic Team"
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, message.as_string())
        st.write(f"Email sent to {email}")
    except Exception as e:
        st.write(f"Failed to send email to {email}: {e}")

# Main Application
def main():
    data = upload_data()
    if data is not None:
        data = preprocess_data(data)
        model, X_train = train_model(data)
        behavioral_analytics(data)
        attendance_heatmap(data)
        explainable_ai(model, X_train)
        leaderboards(data)
        achievement_badges(data)
        send_email_alerts(data)

if __name__ == "__main__":
    main()
