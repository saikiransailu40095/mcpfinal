#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Introduce errors in predictions
def introduce_errors(predictions, error_rate=0):
    n_errors = int(len(predictions) * error_rate)
    error_indices = np.random.choice(len(predictions), n_errors, replace=False)
    for idx in error_indices:
        predictions[idx] = np.random.choice(np.delete(np.unique(predictions), predictions[idx]))
    return predictions

# Train and evaluate function
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, inputs, crops, error_rate=0):
    start_time_train = time.time()
    model.fit(X_train, y_train.ravel())
    training_time = time.time() - start_time_train

    start_time_pred = time.time()
    y_pred = introduce_errors(model.predict(X_test), error_rate=error_rate)
    prediction_time = time.time() - start_time_pred

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    crop_pred = crops[int(model.predict(inputs)[0])]

    st.success(f"Predicted crop ({model_name}): {crop_pred}")
    st.info(f"{model_name} Accuracy: {accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1 Score: {f1 * 100:.2f}%")
    st.info(f"Training Time: {training_time * 1000:.4f} ms")
    st.info(f"Prediction Time: {prediction_time * 1000:.4f} ms")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, crops)

# Plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

# Main application
def main():
    st.title("Crop Prediction Application")
    st.write("This application predicts the best crop for given soil parameters using various machine learning models.")

    # Load dataset
    file_path = "large_data.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found at path: {file_path}")
        return
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return

    # Preprocess dataset
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    crops = df.iloc[:, -1].unique()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # User input for prediction
    st.sidebar.header("Input Soil Parameters")
    inputs = np.array([[
        st.sidebar.number_input(f"Enter value for {col}", value=0.0) for col in df.columns[:-1]
    ]])

    # Model dictionary
    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "MLP": MLPClassifier(random_state=42, max_iter=500, learning_rate_init=0.01),
        "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
    }

    # Buttons for each model
    for model_name, model in models.items():
        if st.button(f"Predict {model_name}"):
            train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, inputs, crops)

if __name__ == "__main__":
    main()


# In[ ]:




