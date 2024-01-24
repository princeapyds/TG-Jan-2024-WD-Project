
from flask import Flask,request
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

@app.route("/train")
def train():
    df = pd.read_excel("D://Users//ds project//False Alarm Cases.xlsx")
    df.drop(["Case No.", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10"], axis=1, inplace=True)
    logr = LogisticRegression()
    logr.fit(df.drop('Spuriosity Index(0/1)',axis=1),
                                                 df['Spuriosity Index(0/1)'])
    joblib.dump(logr,'train.pkl')

    return "Model trained successfully..."



@app.route("/predict",methods=['POST'])
def predict():
    pkl_file = joblib.load('train.pkl')
    data = request.get_json()
    test_data = np.array([data["Ambient Temperature"], data["Calibration"] , data["Unwanted substance deposition"],
    data["Humidity"] , data["H2S Content"], data["detected by"]]).reshape(1,6) # reshpae kel karan reshape array mande aahe

    y_pred = pkl_file.predict(test_data)

    if y_pred == 1:
        return "False Alarm, No Danger"
    else:
        return "True Alarm, Danger "



app.run(port=5000)