# College-Attendance-Class-Performance-Predictor
A predictive tool designed to identify students at academic risk by analyzing patterns in attendance, assignment submissions, and performance metrics. This tool enables early interventions and provides actionable insights to improve student outcomes.

## 🌟 **Features**
- **Input Data**:
  - Attendance records
  - Assignment submission status and marks
  - Midterm/test scores
  - Optional: Participation, quiz scores, and more.
- **Output Predictions**:
  - End-semester performance (grade/score).
  - At-risk student classification: Safe / At-Risk.
  - Personalized suggestions: e.g., “Increase attendance by X% to reach a safe zone.”

### Advanced Features
- **Risk Monitoring Dashboard**:
  - Real-time alerts for students nearing the risk threshold.
  - Risk distribution (Safe vs. At-Risk).
- **Behavioral Insights**:
  - Correlation analysis between attendance, assignments, and grades.
  - Clustering students based on study behavior.
- **Visualizations**:
  - Attendance heatmap.
  - Performance trends: Attendance vs. Grades.
  - Correlation heatmaps.
- **Explainable AI**:
  - Feature importance using SHAP.
- **Leaderboards & Gamification**:
  - Top performers in grades and attendance.
  - Achievement badges for milestones (e.g., perfect attendance).
- **Automated Email Alerts**:
  - Notifications for at-risk students.
- **Custom Reports**:
  - Generate student-level summaries and export them to PDF.

### Machine Learning
- **Classification**:
  - Logistic Regression, Random Forest: Safe vs. At-Risk.
- **Regression**:
  - Linear Regression, XGBoost: Predict exact grades/scores.
- **Clustering**:
  - K-Means: Segment students by study behavior.

## 🛠️ **Tech Stack**
- **Programming Language**: Python
- **Libraries**:
  - **Data Handling & Preprocessing**: Pandas, NumPy
  - **Machine Learning**: Scikit-learn, XGBoost
  - **Visualization**: Matplotlib, Seaborn, SHAP
- **Frontend/UI**: Streamlit
- **Email Integration**: smtplib
- **PDF Generation**: ReportLab (optional)
- **Deployment**: Compatible with cloud platforms like AWS, Azure, or Google Cloud.
