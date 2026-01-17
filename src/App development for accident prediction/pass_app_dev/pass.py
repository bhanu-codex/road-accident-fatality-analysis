# Flask Application Code
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os
# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load pre-trained models
with open('arima_model_pass.pkl', 'rb') as file:
    arima_model = pickle.load(file)  # Load the fitted ARIMA model

with open('gbm_model_pass.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

# Load LSTM model using TensorFlow's load_model
lstm_model = load_model('lstm_model_pass.h5') if os.path.exists('lstm_model_pass.h5') else None

# Updated feature names
features = ['Passengers_PersonsInjured_SeriouslyInjured', 'Passengers_PersonsInjured_MinorInjured']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print('Received data:', data)
        input_features = pd.DataFrame(data, index=[0])
        print('Input features DataFrame:', input_features)

        # Prepare data for models

        # ARIMA model: Use the forecast method on the fitted model
        arima_pred = arima_model.forecast(steps=1).iloc[0]
        print('ARIMA prediction:', arima_pred)

        # Extract only the relevant features for the GBM model
        gbm_features = input_features[features]
        gbm_pred = gbm_model.predict(gbm_features)[0]
        print('GBM prediction:', gbm_pred)

        # Prepare the data for LSTM - reshape to match LSTM input shape
        # LSTM expects input in shape (batch_size, time_steps, features)
        time_steps = 3
        lstm_input_data = np.tile(input_features[features].values, (time_steps, 1)).reshape((1, time_steps, len(features)))

        # Predict with LSTM
        if lstm_model:
            lstm_pred = lstm_model.predict(lstm_input_data)[0][0]
            print('LSTM prediction:', lstm_pred)
        else:
            lstm_pred = 0  # Default value if model is not available

        # Combine predictions
        ensemble_pred = (arima_pred + gbm_pred + lstm_pred) / 3
        print('Ensemble prediction:', ensemble_pred)

        # Find the closest rank and corresponding state from dataset
        df = pd.read_excel("C:/Users/nagas/Downloads/Accidents Classified according to Non-Use of Safety Device ( Non-Wearing of Helmet) by Vict.xlsx")
        closest_rank = df.iloc[(df['Passengers_PersonsDead_Rank'] - ensemble_pred).abs().argsort()[:1]]
        predicted_state = str(closest_rank['States/Uts'].values[0])
        print('Predicted state:', predicted_state)
        predicted_rank = int(closest_rank['Passengers_PersonsDead_Rank'].values[0])
        print('Predicted rank:', predicted_rank)

        return jsonify({'Predicted_Passengers_PersonsKilled_Rank': predicted_rank, 'Predicted_State': predicted_state})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  

