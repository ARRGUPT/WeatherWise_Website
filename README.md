WeatherWise.

WeatherWise is a dynamic weather-based advice system designed for both normal users and farmers. By analyzing weather parameters like humidity, temperature, wind speed, and rainfall, the system provides personalized advice to help users make informed decisions.

1. Features 

API Endpoints: 

/predict: Accepts weather data as input (POST request) and returns personalized advice. 

/: A simple GET endpoint to check if the server is live. 

Error Handling: Ensures robustness with proper input validation and error messages. 

Frontend: A user-friendly interface for entering weather data and receiving advice. 

Live Server: Integrated with ngrok to expose the application publicly.



2. Technology Stack

Backend: Python, Flask 

Frontend: HTML, TailwindCSS, JavaScript 

Libraries: 

Data Analysis: pandas, NumPy 

Data Visualization: Matplotlib, Seaborn 

Machine Learning: scikit-learn (K-Nearest Neighbors) 

Server: ngrok


3. Dataset and Model 

Dataset: Approx. 300 KB in size with key features: humidity, temperature, wind speed, rainfall, and advice. 

Model: K-Nearest Neighbors (KNN) used for prediction. 

Preprocessing: One-hot encoding for the "advice" column. Scaling of numerical features using StandardScaler.


4. API Endpoints /predict (POST Request) Description: Accepts weather data as input and returns advice.


Input Format (JSON): { 
"humidity": 78.2, 
"temperature": 18.3, 
"wind_speed": 21.1, 
"rainfall": 3.1 
} 


Response Format: { "advice": "Weather is moderate, have a good day!" }


/ (GET Request) Description: Checks if the server is live. Response: { "msg": "Server is live" }


5. Installation and Usage Prerequisites Python 3.12.4 Required libraries: pip install numpy pandas matplotlib seaborn scikit-learn Flask pyngrok


6. How to Use the Frontend Open the provided frontend in any modern web browser.


Enter values for humidity, temperature, wind speed, and rainfall. Click "Advice" to receive personalized recommendations.


7. Acknowledgments Tools and Libraries: Python, Flask, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, ngrok, TailwindCSS. 


8. Support: Special thanks to the online development community for their guidance.


9. Contact :

GitHub: https://github.com/ARRGUPT 

Email: guptafamily3005@gmail.com 

LinkedIn: https://www.linkedin.com/in/aryan-gupta-635965302/
