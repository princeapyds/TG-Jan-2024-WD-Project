
import pandas as pd
df = pd.read_excel("C://Users//sanskruti//OneDrive//Documents//data analytics//ds project//False Alarm Cases.xlsx")

print(df.head())

from flask import Flask,request
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__) #name-current module meanss in our case dsproject1

@app.route("/train") #training url chy basis var keli
def train():
    df = pd.read_excel("C://Users//sanskruti//OneDrive//Documents//data analytics//ds project//False Alarm Cases.xlsx") #file pass keli
    df.drop(["Case No.", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10"], axis=1, inplace=True) #data clean
    logr = LogisticRegression() #100% data ni train kel
    logr.fit(df.drop('Spuriosity Index(0/1)',axis=1),
                                                 df['Spuriosity Index(0/1)']) #ithe fit through train kel

    joblib.dump(logr,'train.pkl') #everytime i dont have to train model so i capture it in pickle file
    #kya dump karna hai and kaha dump krna hai

    return "Model trained successfully..." #frontend la msg display karel
app.run(port=5000)


@app.route("/predict",methods=['POST']) #post ke liye frontend se data aayega to vo reqsuest import kiya
def predict():
    pkl_file = joblib.load('train.pkl')
    data = request.get_json() #feilds ithun get.json madhe aale
    test_data = np.array([data["Ambient Temperature"], data["Calibration"] , data["Unwanted substance deposition"],
    data["Humidity"] , data["H2S Content"], data["detected by"]]).reshape(1,6) #ek ek feilds la call kela and list madhe convert kel
    # and reshape kel karan list madhe reshpae naste so array madhe convert kel

    y_pred = pkl_file.predict(test_data) #1row with 6colmn ,Tset_data supplied for prediction and o/p of predction wechecked based on this excel file

    if y_pred == 1:
        return "False Alarm, No Danger"
    else:
        return "True Alarm, Danger "


app.run(port=5001)
