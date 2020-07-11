# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:22:23 2020

@author: Shubhangi sakarkar
"""

from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle

loaded_model=pickle.load(open('random_forest_regression_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('real_2018.csv')
    myprediction=loaded_model.predict(df.iloc[:,:-1].values)
    myprediction=myprediction.tolist()
    return render_template('result.html',prediction=myprediction)



if __name__ =='__main__':
    app.run(debug=True)