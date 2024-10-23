from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('loan_status_model.pkl', 'rb'))

def preprocess_features(features):
    fill_values = {
        'Gender': features['Gender'].mode()[0],
        'Married': features['Married'].mode()[0],
        'Dependents': features['Dependents'].mode()[0],
        'Self_Employed': features['Self_Employed'].mode()[0],
        'LoanAmount': features['LoanAmount'].mean(),
        'Loan_Amount_Term': features['Loan_Amount_Term'].mean(),
        'Credit_History': features['Credit_History'].mode()[0]
    }

    features.fillna(fill_values, inplace=True)

    features.replace({
        'Married': {'No': 0, 'Yes': 1},
        'Gender': {'Male': 1, 'Female': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Education': {'Graduate': 1, 'Not Graduate': 0}
    }, inplace=True)

    features.replace({'Dependents': {'3+': 4}}, inplace=True)

    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()

        # Convert JSON to DataFrame
        features = pd.DataFrame([data])

        # Preprocess the features
        processed_data = preprocess_features(features)

        # Make prediction using the model
        prediction = model.predict(processed_data)

        # Output result based on prediction
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return jsonify({"prediction_text": f"Your loan is likely to be {result}."})

    return jsonify({"error": "Invalid input"}), 400

if __name__ == "__main__":
    app.run(debug=True)
