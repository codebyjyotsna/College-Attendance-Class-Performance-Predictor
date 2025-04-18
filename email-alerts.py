import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

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
