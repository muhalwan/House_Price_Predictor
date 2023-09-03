# app.py

from flask import Flask, render_template, request
import pickle
from sklearn.datasets import fetch_california_housing

app = Flask(__name__)

# Load pre-trained model and scaler
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Fetch data (only to get feature names, data itself is not used here)
california_data = fetch_california_housing()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get data from the form
        data = [float(request.form[key]) for key in california_data.feature_names]
        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)[0]
    return render_template('index.html', prediction=prediction, feature_names=california_data.feature_names)

if __name__ == '__main__':
    app.run(debug=True)
