from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import shap
import streamlit as st

def train_model(data):
    st.subheader("Model Training")
    X = data.drop(columns=["final_grade", "risk_status"], errors="ignore")
    y = data["risk_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write("Training Random Forest Classifier...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("Model Evaluation:")
    st.text(classification_report(y_test, y_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    
    return model, X_train

def explainable_ai(model, X_train):
    st.subheader("Explainable AI: Feature Importance using SHAP")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    st.write("Feature Importance Summary Plot:")
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    st.pyplot(plt.gcf())
