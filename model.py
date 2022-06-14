import pandas as pd
import numpy as np
import catboost
import joblib 
import os
import flask

curr_path = os.path.dirname(os.path.abspath(__file__))

catboost = joblib.load(curr_path+"/catboost1.pkl")

print(type(catboost))

def energy_prediction(x_test):
   
    y_hat = catboost.predict(x_test)
   
    return y_hat

