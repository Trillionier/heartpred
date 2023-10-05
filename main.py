import joblib
from flask import Flask, request, render_template
import numpy as np

# Load the trained machine learning model
model = joblib.load(r"C:\Users\Lenovo\OneDrive\Desktop\heart_disea_prediction\hprid.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_result = None

    if request.method == 'POST':
        # Get input data from the form submitted in the HTML template
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chest_pain = int(request.form['chest_pain'])
        resting_bp = int(request.form['resting_bp'])
        cholesterol = int(request.form['cholesterol'])
        fasting_bs = int(request.form['fastingbs'])
        resting_ecg = int(request.form['restingecg'])
        max_hr = int(request.form['maxhr'])
        exercise_angina = int(request.form['exerciseangina'])
        old_peak = float(request.form['oldpeak'])
        st_slope = int(request.form['stslope'])

        # Create a feature vector for prediction
        features = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                              resting_ecg, max_hr, exercise_angina, old_peak, st_slope]])

        # Make a prediction
        prediction = model.predict(features)

        # Format the prediction as a string
        result = "Heart Disease ditucted" if prediction[0] == 1 else "No Heart Disease"
        print('Prediction Result:', result)

        # Render the result in the HTML template
        return render_template('index.html', prediction_result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

