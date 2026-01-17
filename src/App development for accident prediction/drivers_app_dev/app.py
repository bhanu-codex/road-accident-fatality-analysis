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
with open('arima_modell.pkl', 'rb') as file:
    arima_model = pickle.load(file)  # Load the fitted ARIMA model

with open('gbm_modell.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

# Load LSTM model using TensorFlow's load_model
lstm_model = load_model('lstm_modell.h5') if os.path.exists('lstm_modell.h5') else None

# Updated feature names
features = ['Drivers_PersonsInjured_GrevioslyInjured', 'Drivers_PersonsInjured_MinorInjured']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_features = pd.DataFrame(data, index=[0])

        # Prepare data for models

        # ARIMA model: Use the forecast method on the fitted model
        arima_pred = arima_model.forecast(steps=1).iloc[0]

        # Extract only the relevant features for the GBM model
        gbm_features = input_features[features]
        gbm_pred = gbm_model.predict(gbm_features)[0]

        # Prepare the data for LSTM - reshape to match LSTM input shape
        # LSTM expects input in shape (batch_size, time_steps, features)
        time_steps = 3
        lstm_input_data = np.tile(input_features[features].values, (time_steps, 1)).reshape((1, time_steps, len(features)))

        # Predict with LSTM
        if lstm_model:
            lstm_pred = lstm_model.predict(lstm_input_data)[0][0]
        else:
            lstm_pred = 0  # Default value if model is not available

        # Combine predictions
        ensemble_pred = (arima_pred + gbm_pred + lstm_pred) / 3

        # Find the closest rank and corresponding state from dataset
        df = pd.read_excel("C:/Users/nagas/Downloads/Accidents Classified according to Non-Use of Safety Device ( Non-Wearing of Helmet) by Vict.xlsx")
        closest_rank = df.iloc[(df['Drivers_PersonsDead_Rank'] - ensemble_pred).abs().argsort()[:1]]
        predicted_state = str(closest_rank['States/Uts'].values[0])
        predicted_rank = int(closest_rank['Drivers_PersonsDead_Rank'].values[0])

        return jsonify({'Predicted_Drivers_PersonsKilled_Rank': predicted_rank, 'Predicted_State': predicted_state})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  

