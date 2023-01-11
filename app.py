from math import ceil, floor
from flask import Flask, request, render_template
import pickle
import numpy as np
from numpy import asarray
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
import plotly.express as px
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.ensemble import RandomForestRegressor
app = Flask(__name__, template_folder='templetes')
model = pickle.load(open('SalesModel.pkl', 'rb'))  # loading the model
import randforrest_salesprediction_dz as dz
import randforrest_salesprediction_mc as mc

@app.route('/')
def home():
    return render_template('home.html')
  
@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""

    
    type=str(request.form["type"])
    product_sku = str(request.form["product"])
    
    
    if(type=="dz"):
        prediction=dz.basic(product_sku)
    else:
        prediction=mc.basic(product_sku)    


    # inputs=[[Year1,Month1,value1],[Year2,Month2,value2],[Year3,Month3,value3],[Year4,Month4,value4],[Year5,Month5,value5],[Year6,Month6,value6],[Year7,Month7,value7],[Year8,Month8,value8],[Year9,Month9,value9],[Year10,Month10,value10],[Year11,Month11,value11],[Year12,Month12,value12]]
    # inputs = inputs.flatten()

    # prediction = model.predict(asarray([inputs]))
    prediction_text=prediction
    
    return render_template('index.html', prediction_text=f'{prediction_text}')

if __name__ == "__main__":
    app.run()