import streamlit as st
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model_file = "model.joblib"

# Train a new model with only 4 features
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)[['mean radius', 'mean texture', 'mean perimeter', 'mean area']]
y = data.target

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, model_file)
st.write("Model trained on 4 features and saved.")

# Display model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Streamlit UI for predictions
st.title("Breast Cancer Prediction App")
st.write("Predict the presence of breast cancer based on user input.")

# Input fields (for 4 features)
mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=30.0, step=0.1)
mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=40.0, step=0.1)
mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=200.0, step=0.1)
mean_area = st.number_input("Mean Area", min_value=0.0, max_value=3000.0, step=1.0)

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'mean radius': [mean_radius],
    'mean texture': [mean_texture],
    'mean perimeter': [mean_perimeter],
    'mean area': [mean_area]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction:", "Malignant" if prediction[0] == 0 else "Benign")
