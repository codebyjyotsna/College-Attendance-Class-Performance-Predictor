import streamlit as st
from utils.preprocessing import preprocess_data, upload_data
from utils.ml_models import train_model, explainable_ai
from utils.visualizations import behavioral_analytics, attendance_heatmap, leaderboards
from utils.email_alerts import send_email_alerts
import os

# Main Application
def main():
    st.title("College Attendance + Class Performance Predictor")
    
    # Upload and preprocess data
    data = upload_data()
    if data is not None:
        data = preprocess_data(data)

        # Train the model
        model, X_train = train_model(data)

        # Analyze behavior
        behavioral_analytics(data)
        
        # Visualize attendance heatmap
        attendance_heatmap(data)
        
        # Explainable AI
        explainable_ai(model, X_train)

        # Leaderboards
        leaderboards(data)

        # Send email alerts to at-risk students
        send_email_alerts(data)

if __name__ == "__main__":
    main()
