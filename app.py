from flask import Flask, request, render_template, url_for,request
import pickle,time,os,io,re
import pandas as pd
from sys import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from pandas.plotting import scatter_matrix
from scipy.stats import gaussian_kde
from pandas.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from sklearn import manifold 
from mpl_toolkits.mplot3d import Axes3D
import csv
import scipy
from sklearn import preprocessing
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from string import punctuation, digits
from sklearn.svm import SVC


start_time = time.time()
vec = open("models/SVCmodelVS.pkl", 'rb')
loaded_model=pickle.load(vec)

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("webform.html")

@app.route('/', methods=['POST','GET'])
def home():
    global loaded_model
    columns=["radius_mean", "texture_mean", "perimeter_mean", "area_mean" ,"smoothness_mean" ,"compactness_mean", "concavity_mean" ,"concave points_mean", "symmetry_mean" ,"fractal_dimension_mean"]
    datax= pd.DataFrame(index=["10"], columns=columns) 
    #print(type(datax))
    #print(datax)
    value_radius_mean=request.form["radius_mean"]
    #value_radius_mean=float(value_radius_mean)
    datax.loc[["10"], "radius_mean"]=float(value_radius_mean)
    

    value_texture_mean=request.form["texture_mean"]
    #value_texture_mean=float(value_texture_mean)
    datax.loc[["10"],"texture_mean"]=float(value_texture_mean)

    value_perimeter_mean=request.form["perimeter_mean"]
    #value_perimeter_mean=float(value_perimeter_mean)
    datax.loc[["10"],"perimeter_mean"]=float(value_perimeter_mean)
    
    value_area_mean=request.form["area_mean"]
    #value_area_mean=float(value_area_mean)
    datax.loc[["10"],"area_mean"]=float(value_area_mean)

    value_smoothness_mean=request.form["smoothness_mean"]
    #value_smoothness_mean=float(value_smoothness_mean)
    datax.loc[["10"],"smoothness_mean"]=float(value_smoothness_mean)

    value_compactness_mean=request.form["compactness_mean"]
    #value_compactness_mean=float(value_compactness_mean)
    datax.loc[["10"],"compactness_mean"]=float(value_compactness_mean)

    value_concavity_mean=request.form["concavity_mean"]
    #value_concavity_mean=float(value_concavity_mean)
    datax.loc[["10"],"concavity_mean"]=float(value_concavity_mean)

    value_concave_points_mean=request.form["concave points_mean"]
    #value_concave_points_mean=float(value_concave_points_mean)
    datax.loc[["10"],"concave points_mean"]=float(value_concave_points_mean)

    value_symmetry_mean=request.form["symmetry_mean"]
    #value_symmetry_mean=float(value_symmetry_mean)
    datax.loc[["10"],"symmetry_mean"]=float(value_symmetry_mean)

    value_fractal_dimension_mean=request.form["fractal_dimension_mean"]
    #value_fractal_dimension_mean=float(value_fractal_dimension_mean)
    datax.loc[["10"],"fractal_dimension_mean"]=float(value_fractal_dimension_mean)

    
    # value_radius_mean=float(value_radius_mean)
    # value_texture_mean=float(value_texture_mean)
    # value_perimeter_mean=float(value_perimeter_mean)
    # value_area_mean=float(value_area_mean)
    # value_smoothness_mean=float(value_smoothness_mean)
    # value_compactness_mean=float(value_compactness_mean)
    # value_concavity_mean=float(value_concavity_mean)
    # value_concave_points_mean=float(value_concave_points_mean)
    # value_symmetry_mean=float(value_symmetry_mean)
    # value_fractal_dimension_mean=float(value_fractal_dimension_mean)

    

    # dizi=np.array([[value_radius_mean,value_texture_mean,value_perimeter_mean,value_area_mean,value_smoothness_mean,value_compactness_mean,value_concavity_mean,value_concave_points_mean,value_symmetry_mean,value_fractal_dimension_mean]])
    # print(dizi)
    # dizi=np.array([[19.69 , 21.25, 130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999]])
    # print(dizi)
    
    #print(np.mean(dizi))
    # original_mean=np.mean(dizi)
    # original_std=np.std(dizi)
    #datax=value_radius_mean,value_texture_mean,value_perimeter_mean,value_area_mean,value_smoothness_mean+value_compactness_mean+value_concavity_mean+value_concave_points_mean+value_symmetry_mean+value_fractal_dimension_mean
    
    #print(type(datax))
    print(datax)
    from pickle import load
    scale = load(open('models/scale.pkl', 'rb'))
    datax_std=scale.transform(datax)
    print(type(datax_std))
    print(datax_std)

    # datax_std=[[ 1.5734154 ,  0.39642466,  1.5660444 ,  1.54631148,  0.96172902,
    #       1.13256293,  1.39792404,  2.0412267 ,  0.96338541, -0.38441926]]
    #print(datax_std)
    #print(datax.dtypes)
    #print(type(datax_std))
    #ax=type(datax)
    #pca1=PCA(n_components=10,svd_solver='randomized')
    #pca1.fit(datax_std)
    #datax_std_pca=pca1.transform(datax_std)
    #print(datax)
    predictedx=loaded_model.predict(datax_std)
    result=''
    if predictedx==[1]:
        result='KÖTÜ HUYLU KANSER HÜCRESİ' 
    else:
        result='İYİ HUYLU NODÜL-KANSER HÜCRESİ SAPTANMAMIŞTIR.'
    
    ths=open('C:\\Users\\Berk\\Desktop\\health\\logs.txt',"w")
    ths.write(str(datax))

    return render_template('webform.html',value=result)
	





if __name__=="__main__":
    app.run(debug=True)

