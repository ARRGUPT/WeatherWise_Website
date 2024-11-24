from pyngrok import ngrok, conf
from flask import Flask, request, jsonify

# print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
conf.get_default().auth_token = '2oQp2VUXGtWILvfWSAlmYXLZxNo_47R1erFFXMmnvJAQbs33H'


app = Flask(__name__)

port = "5000"

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

# GET route for server status check
@app.route('/', methods=['GET'])
def home():
    return jsonify({"msg": "Server is live"})

# POST route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    humidity = data['humidity']
    temperature = data['temperature']
    wind_speed = data['wind_speed']
    rainfall = data['rainfall']

    advice_text = get_advice(humidity, temperature, wind_speed, rainfall)
    print(advice_text)

    return jsonify({'Advice': advice_text})

if __name__ == '__main__':
    try:
      public_url = ngrok.connect(port).public_url
      print(public_url)
      app.run(port=port)
    finally:
      ngrok.disconnect(public_url)
