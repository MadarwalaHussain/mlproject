from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import datetime

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipleline, CustomData

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Check if request is JSON (from Postman/API) or form data (from web page)
            if request.is_json:
                # Handle JSON data from Postman/API
                json_data = request.get_json()
                data = CustomData(
                    gender=json_data.get('gender'),
                    race_ethnicity=json_data.get('race_ethnicity'),
                    parental_level_of_education=json_data.get('parental_level_of_education'),
                    lunch=json_data.get('lunch_type'),
                    test_preparation_course=json_data.get('test_preparation_course'),
                    reading_score=float(json_data.get('reading_score', 0)),
                    writing_score=float(json_data.get('writing_score', 0))
                )
            else:
                # Handle form data from web page
                data = CustomData(
                    gender=request.form.get('gender'),
                    race_ethnicity=request.form.get("race/ethnicity"),
                    parental_level_of_education=request.form.get("parental level of education"),
                    lunch=request.form.get("lunch"),
                    test_preparation_course=request.form.get("test preparation course"),
                    reading_score=float(request.form.get("reading score", 0)),
                    writing_score=float(request.form.get("writing score", 0))
                )

            predict_df = data.get_data_as_frame()
            print(predict_df)

            predict_pipeline = PredictPipleline()
            results = predict_pipeline.predict(predict_df)

            # Return JSON response for API calls, HTML for web page
            if request.is_json:
                return jsonify({
                    'success': True,
                    'predicted_math_score': float(results[0]),
                    'message': 'Prediction successful'
                })
            else:
                return render_template('home.html', results=results[0])

        except Exception as e:
            # Handle errors gracefully
            if request.is_json:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'message': 'Prediction failed'
                }), 400
            else:
                return render_template('home.html', error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0")
