import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def behavioral_analytics(data):
    st.subheader("Behavioral Analytics")
    if "assignment_submissions" in data.columns and "final_grade" in data.columns:
        sns.boxplot(data=data, x="assignment_submissions", y="final_grade")
        plt.title("Assignment Submission Frequency vs. Final Grade")
        st.pyplot(plt.gcf())
    else:
        st.write("Required columns not found in the dataset.")

def attendance_heatmap(data):
    st.subheader("Attendance Heatmap")
    if "date" in data.columns and "attendance" in data.columns:
        attendance_pivot = data.pivot_table(index="date", columns="student_id", values="attendance", fill_value=0)
        sns.heatmap(attendance_pivot, cmap="YlGnBu", cbar=True)
        plt.title("Attendance Heatmap")
        st.pyplot(plt.gcf())
    else:
        st.write("Required columns (date, attendance) not found in the dataset.")

def leaderboards(data):
    st.subheader("Leaderboards")
    if "final_grade" in data.columns:
        top_students = data.sort_values(by="final_grade", ascending=False).head(10)
        st.write("Top 10 Students by Final Grade:")
        st.dataframe(top_students[["student_id", "final_grade"]])
