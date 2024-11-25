from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# Allow requests from localhost (127.0.0.1) for local development
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5500", "https://weatherwise-website.onrender.com"]}})



# Load the pre-trained models and encoders
with open('./models/knn_model.pkl', 'rb') as model_file:
    knn_loaded = pickle.load(model_file)
with open('./models/scaler.pkl', 'rb') as scaler_file:
    scaler_loaded = pickle.load(scaler_file)
with open('./models/encoder.pkl', 'rb') as encoder_file:
    encoder_loaded = pickle.load(encoder_file)

app = Flask(__name__)

def get_advice(humidity, temperature, wind_speed, rainfall):
    # Prepare the input data
    input_data = pd.DataFrame(
        [[humidity, temperature, wind_speed, rainfall]],
        columns=['Humidity (%)', 'Temperature (°C)', 'Wind Speed (km/h)', 'Rainfall (mm)']
    )
    # Scale the input data
    input_scaled = scaler_loaded.transform(input_data)

    # Get prediction
    prediction = knn_loaded.predict(input_scaled)

    # Handle different prediction formats
    if len(prediction.shape) == 1:  # Single label prediction
        advice_index = prediction[0]
    else:  # One-hot encoded prediction
        advice_index = np.argmax(prediction)

    try:
        # Retrieve advice text using the encoder
        advice_text = encoder_loaded.categories_[0][advice_index]
    except IndexError:
        print("Error: Advice index out of range.")
        advice_text = "Unknown advice. Check your data and model."

    return advice_text

# GET route for server status check
@app.route('/', methods=['GET'])
def home():
    return jsonify({"msg": "Server is live"})


# POST route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        data = request.get_json(force=True)
        app.logger.debug(f"Received data: {data}")
        humidity = data['humidity']
        temperature = data['temperature']
        wind_speed = data['wind_speed']
        rainfall = data['rainfall']

        # Get advice based on the input data
        advice_text = get_advice(humidity, temperature, wind_speed, rainfall)

        # Return the response as JSON
        return jsonify({'advice': advice_text})
    except Exception as e:
        # Handle exceptions and return error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    # Dynamically set the port for deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
