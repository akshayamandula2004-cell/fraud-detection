from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import logging
import os
from joblib import load

# Initialize Flask app
app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Load models with error handling
try:
    credit_model = load('credit_card_model.pkl')
    logging.info("Credit fraud model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load 'credit_card_model.pkl': {e}")
    credit_model = None

try:
    upi_model = load('upi_model.pkl')
    logging.info("UPI fraud model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load 'upi_model.pkl': {e}")
    upi_model = None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

import pandas as pd  # Add this to your imports

@app.route('/credit_fraud', methods=['GET', 'POST'])
def credit_fraud():
    """Handle credit fraud detection."""
    if request.method == 'POST':
        # Get features input from form
        features_input = request.form.get('features', '').strip()
        if not features_input:
            return render_template(
                'credit.html',
                error="Features cannot be empty.",
                features_input=features_input
            )

        try:
            # Convert input string to float list
            features = [float(x) for x in features_input.split(',')]

            # Expecting 30 total: remove first one if it's 'Time'
            if len(features) != 30:
                raise ValueError("Exactly 30 features are required.")
            features = features[1:]  # Drop 'Time' column (first value)

            if len(features) != 29:
                raise ValueError("Expected 29 features (V1-V28 + Amount) after removing 'Time'.")

            # Define correct feature names (without 'Time')
            credit_feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']

            # Create DataFrame
            features_df = pd.DataFrame([features], columns=credit_feature_names)

            # Check if model is available
            if credit_model is None:
                raise RuntimeError("Credit fraud model is unavailable.")

            # Make prediction
            prediction = credit_model.predict(features_df)
            result = (
                "Fraudulent transaction detected"
                if prediction[0] == 1 else
                "Not a fraudulent transaction"
            )

            return render_template('result.html', title="Credit Fraud Result", result=result)

        except ValueError as ve:
            logging.error(f"Invalid input: {ve}")
            return render_template(
                'credit.html',
                error=str(ve),
                features_input=features_input
            )
        except Exception as e:
            logging.error(f"Error during credit fraud prediction: {e}")
            return render_template(
                'credit.html',
                error="An unexpected error occurred during prediction.",
                features_input=features_input
            )

    return render_template('credit.html')

@app.route('/upi_fraud', methods=['GET', 'POST'])
def upi_fraud():
    if request.method == 'POST':
        try:
            withdrawal = float(request.form.get('Withdrawal', '0').strip())
            deposit = float(request.form.get('Deposit', '0').strip())
            balance = float(request.form.get('Balance', '0').strip())

            # Define expected feature names
            upi_feature_names = ['Withdrawal', 'Deposit', 'Balance']
            input_df = pd.DataFrame([[withdrawal, deposit, balance]], columns=upi_feature_names)

            if upi_model is None:
                raise RuntimeError("UPI fraud model is unavailable.")

            prediction = upi_model.predict(input_df)
            result = "Fraudulent UPI transaction detected" if prediction[0] == 1 else "Not a fraudulent UPI transaction"
            return render_template('result.html', title="UPI Fraud Result", result=result)

        except ValueError:
            logging.error("Invalid input: Non-numeric values provided.")
            return render_template(
                'upi.html',
                error="Invalid input. Please enter numeric values.",
                withdrawal=request.form.get('Withdrawal', ''),
                deposit=request.form.get('Deposit', ''),
                balance=request.form.get('Balance', '')
            )
        except Exception as e:
            logging.error(f"Error during UPI fraud prediction: {e}")
            return render_template(
                'upi.html',
                error="An unexpected error occurred during prediction.",
                withdrawal=request.form.get('Withdrawal', ''),
                deposit=request.form.get('Deposit', ''),
                balance=request.form.get('Balance', '')
            )

    return render_template('upi.html')

# Run Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
