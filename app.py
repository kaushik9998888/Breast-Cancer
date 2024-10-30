# Import necessary libraries
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import subprocess
import sys

# Install joblib if not present
try:
    import joblib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib




# Specify the model file name
model_file = "model.joblib"

# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)[['mean radius', 'mean texture', 'mean perimeter', 'mean area']]
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, model_file)
print("Model trained on 4 features and saved as", model_file)

# Calculate model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Streamlit UI for predictions
def run_streamlit_app():
    st.title("Breast Cancer Prediction App")
    st.write("Predict the presence of breast cancer based on user input.")

    # Input fields for prediction
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

    # Load the model for predictions
    model = joblib.load(model_file)

    # Prediction button
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.write("Prediction:", "The person is dead" if prediction[0] == 0 else "The person is living")

# Uncomment the following line to run the Streamlit app in a new terminal
# run_streamlit_app()

# If you want to run the Streamlit app, execute the script in a terminal with: 
# streamlit run your_script_name.py
