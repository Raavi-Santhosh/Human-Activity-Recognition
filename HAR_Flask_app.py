from flask import Flask, render_template, request,jsonify
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# class_labels = ['Walking','Walking_UP',"Walking_DOWN","Sitting","Standing","Laying"]


app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))

@app.route("/",methods = ["POST","GET"])
def home():
    if request.method == "GET":
        display=False
        return render_template("home.html",display=True,mylist=["NULL"])
        
    elif request.method == "POST":
        file = request.files["Browse"]
        des = "/".join([app_root,file.filename])
        file.save(des)
        sample = pd.read_csv(des)
        
        with open("linear_svm_model.pkl",'rb') as file:
            model = pickle.load(file)
            preds = model.predict(sample)
        
        return render_template("home.html",display = True,mylist=preds)

@app.route("/predict",methods=["POST"])
def upload():
    file = request.files["Browse"]
    des = "/".join([app_root,file.filename])
    file.save(des) 
    return "file uploaded"
if __name__ == '__main__':
    app.run()
