import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# App title
st.title("📊 Logistic Regression Demo")

# Sample dataset (Student Pass/Fail)
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

st.subheader("📌 Dataset")
st.write(df)

# Features and target
X = df[["Hours_Studied"]]
y = df["Pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📈 Model Evaluation")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# User input
st.subheader("🔮 Predict Pass or Fail")
hours = st.number_input("Enter study hours:", min_value=0.0, max_value=24.0)

if st.button("Predict"):
    result = model.predict([[hours]])
    if result[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")
