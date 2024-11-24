
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
# %matplotlib inline


df = pd.read_csv('./api/WeatherWise_Dataset.csv', encoding='latin-1')

df.head()

df.info()

df.isnull().sum()

df.describe()

# Histogram

for col in ['Humidity (%)', 'Temperature (°C)', 'Wind Speed (km/h)', 'Rainfall (mm)']:
  sns.histplot(df[col], kde=True)
  plt.title(f"Distribution of {col}")
  plt.show()

df.info()

# heatmap for correlation matrix

numeric_data = df.select_dtypes(include=['float64'])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix/Heatmap")
plt.show()

# Relationship(boxplot) b/w features & advice

for col in ['Humidity (%)', 'Temperature (°C)', 'Wind Speed (km/h)', 'Rainfall (mm)']:
  sns.boxplot(y='Advice', x=col, data=df)
  plt.title(f"Advice vs {col}")
  plt.show()

# countplot of advice

sns.countplot(y='Advice', data=df)
plt.title("Count of Advice")
plt.show()

# Encoding and concatination of Advice

encoder = OneHotEncoder()

# Encode advice
advice_encoded = encoder.fit_transform(df[['Advice']]).toarray()
advice_df = pd.DataFrame(advice_encoded, columns=encoder.get_feature_names_out(['Advice']))

# Concatination of advice
df_encoded = pd.concat([df.drop('Advice', axis=1), advice_df],axis=1)

df_encoded.head()

# splitting for model training/testing

from sklearn.model_selection import train_test_split

X = df_encoded[['Humidity (%)', 'Temperature (°C)', 'Wind Speed (km/h)', 'Rainfall (mm)']]
y = df_encoded.drop(['Humidity (%)', 'Temperature (°C)', 'Wind Speed (km/h)', 'Rainfall (mm)'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

X_train.head()

y_train.head()

# scaling the features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled[:5]

# Training the model by knn

k = 7                                                   # we can adjust this value for better performance/pridiction
knn = KNeighborsClassifier(n_neighbors=k)               # initializaton

knn.fit(X_train_scaled, y_train)                        # train the model on training data
y_pred = knn.predict(X_test_scaled)                     # prediction on test data

# Evaluation of model
print("Accuracy score:", accuracy_score(y_test, y_pred))

print("\nClassification report:", classification_report(y_test, y_pred))

# creating a function to get advice

def get_advice(humidity, temperature, wind_speed, rainfall):
  input_data = pd.DataFrame(
        [[humidity, temperature, wind_speed, rainfall]],
        columns=['Humidity (%)', 'Temperature (°C)', 'Wind Speed (km/h)', 'Rainfall (mm)'])          # input data
  input_scaled = scaler.transform(input_data)                                     # input data is scaled
  prediction = knn.predict(input_scaled)                                          # prediction

  if len(prediction.shape) == 1:                                                  # Single label prediction
    advice_index = prediction[0]
  else:                                                                           # One-hot encoded prediction
    advice_index = np.argmax(prediction)                                          # get the index of predicted advice

  try:
    advice_text = encoder.categories_[0][advice_index]                            # This will be the advice
  except IndexError:
    print("Error: Advice index out of range.")
    advice_text = "Unknown advice. Check your data and model."

  return advice_text

# Example usage

import json as json
humidity = 78.2
temperature = 18.3
wind_speed = 21.1
rainfall = 3.1
predicted_advice = get_advice(humidity, temperature, wind_speed, rainfall)
print("Predicted Advice:", predicted_advice)
print(type(predicted_advice))

# Testing

test_cases = [
    [78.2, 18.3, 21.1, 3.1],
    [66.6, 12.4, 5, 7],
    [82.9, 5.7, 23.9, 9.4],
    [87.9, 27.2, 11.3, 2.5]
]

for case in test_cases:
  print(f"Input: {case}")
  print(f"Predicted Advice: {get_advice(*case)}")
  print("-------------------------------")

import pickle

# save model,scaler & encoder
with open('knn_model.pkl', 'wb') as model_file:
  pickle.dump(knn, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
  pickle.dump(scaler, scaler_file)
with open('encoder.pkl', 'wb') as encoder_file:
  pickle.dump(encoder, encoder_file)

# load these objects

with open('knn_model.pkl', 'rb') as model_file:
  knn_loaded = pickle.load(model_file)
with open('scaler.pkl', 'rb') as model_file:
  scaler_loaded = pickle.load(model_file)
with open('encoder.pkl', 'rb') as model_file:
  encoded_loader = pickle.load(model_file)

# from google.colab import files
# files.download('knn_model.pkl')
# files.download('scaler.pkl')
# files.download('encoder.pkl')