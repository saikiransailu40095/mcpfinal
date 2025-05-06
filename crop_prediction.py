import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Load dataset
engine_data = pd.read_csv('engine_data (1).csv')

# Data preprocessing steps (example - adjust based on actual notebook content)
# Assuming 'Condition' is the target
X = engine_data.drop('Condition', axis=1)
y = engine_data['Condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Evaluation
predictions = model.predict(X_test_scaled)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

# Save the model
with open('hhmodel.pkl', 'wb') as f:
    pickle.dump(model, f)

# (END of notebook code)

# --- Code from app.py ---

import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('hhmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Customized feature ranges
custom_ranges = {
    'Engine rpm': (61.0, 2239.0),
    'Lub oil pressure': (0.003384, 7.265566),
    'Fuel pressure': (0.003187, 21.138326),
    'Coolant pressure': (0.002483, 7.478505),
    'lub oil temp': (71.321974, 89.580796),
    'Coolant temp': (61.673325, 195.527912),
    'Temperature_difference': (-22.669427, 119.008526)
}

# Feature descriptions
feature_descriptions = {
    'Engine rpm': 'Revolution per minute of the engine.',
    'Lub oil pressure': 'Pressure of the lubricating oil.',
    'Fuel pressure': 'Pressure of the fuel.',
    'Coolant pressure': 'Pressure of the coolant.',
    'lub oil temp': 'Temperature of the lubricating oil.',
    'Coolant temp': 'Temperature of the coolant.',
    'Temperature_difference': 'Temperature difference between components.'
}

def main():
    st.title("Engine Condition Prediction")
    
    st.sidebar.title("Feature Descriptions")
    for feature, description in feature_descriptions.items():
        st.sidebar.markdown(f"**{feature}:** {description}")
    
    engine_rpm = st.slider("Engine RPM", min_value=float(custom_ranges['Engine rpm'][0]),
                           max_value=float(custom_ranges['Engine rpm'][1]),
                           value=float(custom_ranges['Engine rpm'][1] / 2))
    lub_oil_pressure = st.slider("Lub Oil Pressure", min_value=custom_ranges['Lub oil pressure'][0],
                                 max_value=custom_ranges['Lub oil pressure'][1],
                                 value=(custom_ranges['Lub oil pressure'][0] + custom_ranges['Lub oil pressure'][1]) / 2)
    fuel_pressure = st.slider("Fuel Pressure", min_value=custom_ranges['Fuel pressure'][0],
                              max_value=custom_ranges['Fuel pressure'][1],
                              value=(custom_ranges['Fuel pressure'][0] + custom_ranges['Fuel pressure'][1]) / 2)
    coolant_pressure = st.slider("Coolant Pressure", min_value=custom_ranges['Coolant pressure'][0],
                                 max_value=custom_ranges['Coolant pressure'][1],
                                 value=(custom_ranges['Coolant pressure'][0] + custom_ranges['Coolant pressure'][1]) / 2)
    lub_oil_temp = st.slider("Lub Oil Temperature", min_value=custom_ranges['lub oil temp'][0],
                             max_value=custom_ranges['lub oil temp'][1],
                             value=(custom_ranges['lub oil temp'][0] + custom_ranges['lub oil temp'][1]) / 2)
    coolant_temp = st.slider("Coolant Temperature", min_value=custom_ranges['Coolant temp'][0],
                             max_value=custom_ranges['Coolant temp'][1],
                             value=(custom_ranges['Coolant temp'][0] + custom_ranges['Coolant temp'][1]) / 2)
    temp_difference = st.slider("Temperature Difference", min_value=custom_ranges['Temperature_difference'][0],
                                max_value=custom_ranges['Temperature_difference'][1],
                                value=(custom_ranges['Temperature_difference'][0] + custom_ranges['Temperature_difference'][1]) / 2)
    
    if st.button("Predict Engine Condition"):
        result, confidence = predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure,
                                              coolant_pressure, lub_oil_temp, coolant_temp, temp_difference)
        if result == 0:
            st.info(f"The engine is predicted to be in a normal condition. Confidence: {1.0 - confidence:.2%}")
        else:
            st.warning(f"Warning! Please investigate further. Confidence: {1.0 - confidence:.2%}")
    
    if st.button("Reset Values"):
        st.experimental_rerun()

def predict_condition(engine_rpm, lub_oil_pressure, fuel_pressure,
                      coolant_pressure, lub_oil_temp, coolant_temp, temp_difference):
    input_data = np.array([engine_rpm, lub_oil_pressure, fuel_pressure,
                           coolant_pressure, lub_oil_temp, coolant_temp, temp_difference]).reshape(1, -1)
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[:, 1]
    return prediction[0], confidence[0]

if __name__ == "__main__":
    main()
