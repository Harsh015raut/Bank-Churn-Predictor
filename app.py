from flask import Flask,request,render_template 
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler 
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

#Route Home Page
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            CreditScore=int(request.form.get('CreditScore')),
            Geography=request.form.get('Geography'),
            Gender=request.form.get('Gender'),
            Age=int(request.form.get('Age')),
            Tenure=int(request.form.get('Tenure')),
            Balance=int(request.form.get('Balance')),
            HasCrCard=int(request.form.get('HasCrCard')) if request.form.get('HasCrCard') is not None else 0,
            IsActiveMember=int(request.form.get('IsActiveMember')) if request.form.get('IsActiveMember') is not None else 0,
            EstimatedSalary=int(request.form.get('EstimatedSalary'))
        )

        pred_df = data.get_data_as_df()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        if results[0] == 0:
            results_str = 'No'
        else:
            results_str = 'Yes'
        return render_template('home.html', results=results_str)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) 


