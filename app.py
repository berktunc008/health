from fileinput import filename
from imp import load_module
from multiprocessing.sharedctypes import Value
from unittest import result
from flask import Flask, request, render_template, url_for,request
import pickle
import time
import io
import re
import pandas as pd
from sys import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors
from matplotlib.pyplot import imshow
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
from pickle import load
from keras.models import load_model
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

app=Flask(__name__)



@app.route("/")
def index():
    return render_template("home.html")

@app.route("/home" , methods=['GET'])
def home():
    return render_template("home.html")

@app.route("/breast")
def breast():
    return render_template("breastcancer.html")

#Template heart oluşturulunca heart.html olarak değişilecek.
@app.route("/heart")
def heart():
    return render_template("heartattack.html")

@app.route("/brain")
def brain():
    return render_template("braintumor.html")

@app.route("/hakkimizda")
def hakkimizda():
    return render_template("hakkimizda.html")    

@app.route('/breast', methods=['POST','GET'])
def breastcancer():
    #Model load edildi.
    model_breastcancer = load(open("models/model_breastcancer/SVCmodelVS.pkl", 'rb'))

    #Dataframe atamaları için columns oluşturma işlemi yapıldı.   
    columns=["radius_mean", "texture_mean", "perimeter_mean", "area_mean" ,"smoothness_mean" ,"compactness_mean", "concavity_mean" ,"concave points_mean", "symmetry_mean" ,"fractal_dimension_mean"]
    datax= pd.DataFrame(index=["10"], columns=columns)

    #Breastcancer için formdan gelen veriler dataframe'e aktarıldı.
    value_radius_mean=request.form["radius_mean"]
    datax.loc[["10"], "radius_mean"]=float(value_radius_mean)
    
    value_texture_mean=request.form["texture_mean"]
    datax.loc[["10"],"texture_mean"]=float(value_texture_mean)

    value_perimeter_mean=request.form["perimeter_mean"]
    datax.loc[["10"],"perimeter_mean"]=float(value_perimeter_mean)
    
    value_area_mean=request.form["area_mean"]
    datax.loc[["10"],"area_mean"]=float(value_area_mean)

    value_smoothness_mean=request.form["smoothness_mean"]
    datax.loc[["10"],"smoothness_mean"]=float(value_smoothness_mean)

    value_compactness_mean=request.form["compactness_mean"]
    datax.loc[["10"],"compactness_mean"]=float(value_compactness_mean)

    value_concavity_mean=request.form["concavity_mean"]
    datax.loc[["10"],"concavity_mean"]=float(value_concavity_mean)

    value_concave_points_mean=request.form["concave points_mean"]
    datax.loc[["10"],"concave points_mean"]=float(value_concave_points_mean)

    value_symmetry_mean=request.form["symmetry_mean"]
    datax.loc[["10"],"symmetry_mean"]=float(value_symmetry_mean)

    value_fractal_dimension_mean=request.form["fractal_dimension_mean"]
    datax.loc[["10"],"fractal_dimension_mean"]=float(value_fractal_dimension_mean)

    

    # dizi=np.array([[value_radius_mean,value_texture_mean,value_perimeter_mean,value_area_mean,value_smoothness_mean,value_compactness_mean,value_concavity_mean,value_concave_points_mean,value_symmetry_mean,value_fractal_dimension_mean]])
    # dizi=np.array([[19.69 , 21.25, 130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999]])
       
    #datax=value_radius_mean,value_texture_mean,value_perimeter_mean,value_area_mean,value_smoothness_mean+value_compactness_mean+value_concavity_mean+value_concave_points_mean+value_symmetry_mean+value_fractal_dimension_mean

    #Model içerisinde gerçekleştirilen StandardScaler işlemi load edildi.
    scale = load(open('models/model_breastcancer/scale.pkl', 'rb'))
    datax_std=scale.transform(datax)
    print(datax_std)

    # datax_std=[[ 1.5734154 ,  0.39642466,  1.5660444 ,  1.54631148,  0.96172902,
    #       1.13256293,  1.39792404,  2.0412267 ,  0.96338541, -0.38441926]]s
    #print(datax_std)
    #print(datax.dtypes)
    #print(type(datax_std))
    #ax=type(datax)
    #pca1=PCA(n_components=10,svd_solver='randomized')
    #pca1.fit(datax_std)
    #datax_std_pca=pca1.transform(datax_std)
    #print(datax)

    #Modelin kullanılması ile formdan gelen veriler üzerinde tahmin yapılıyor.
    predictedx=model_breastcancer.predict(datax_std)
    result=''
    filenames=['']
    if predictedx==[1]:
        #result='KÖTÜ HUYLU KANSER HÜCRESİ-KANSER HÜCRESİ SAPTANMIŞTIR'
        filenames=['sad_doctorv2.jpg']
    elif predictedx==[0]:
        #result='İYİ HUYLU NODÜL-KANSER HÜCRESİ SAPTANMAMIŞTIR.'
        filenames=['HappyDoctorV2.jpg']

    #ths=open('C:\\Users\\Berk\\Desktop\\health\\logs.txt',"w")
    #ths.write(str(datax))

    return render_template('breastcancer.html',value=result, filenames=filenames)







@app.route('/heart', methods=['POST','GET'])
def heart_disease():
  #Heart Disease Model yüklendi
  
    model_heart = load(open("models/model_heart/logisticreg_heart.pkl", 'rb'))

    columns=["BMI", "Smoking", "AlcoholDrinking", "Stroke" ,"PhysicalHealth" ,"MentalHealth", "DiffWalking" ,"Sex", "AgeCategory" ,"Race","Diabetic","PhysicalActivity","GenHealth","SleepTime","Asthma","KidneyDisease","SkinCancer"]
    dataheart= pd.DataFrame(index=["X"], columns=columns)
    print(dataheart)
    BMI=request.form["BMI"]
    dataheart.loc[["X"], "BMI"]=BMI

    Smoking=request.form["Smoking"]
    if(Smoking=="Evet"):
        Smoking=1
    elif(Smoking=="Hayır"):
        Smoking=0
    dataheart.loc[["X"], "Smoking"]=Smoking


    AlcoholDrinking=request.form["AlcoholDrinking"]
    if(AlcoholDrinking=="Evet"):
        AlcoholDrinking=1
    elif(AlcoholDrinking=="Hayır"):
        AlcoholDrinking=0
    dataheart.loc[["X"], "AlcoholDrinking"]=AlcoholDrinking


    Stroke=request.form["Stroke"]
    if(Stroke=="Evet"):
        Stroke=1
    elif(Stroke=="Hayır"):
        Stroke=0
    dataheart.loc[["X"], "Stroke"]=Stroke



    PhysicalHealth=request.form["PhysicalHealth"]
    dataheart.loc[["X"], "PhysicalHealth"]=PhysicalHealth



    MentalHealth=request.form["MentalHealth"]
    dataheart.loc[["X"], "MentalHealth"]=MentalHealth



    DiffWalking=request.form["DiffWalking"]
    if(DiffWalking=="Evet"):
        DiffWalking=1
    elif(DiffWalking=="Hayır"):
        DiffWalking=0
    dataheart.loc[["X"], "DiffWalking"]=DiffWalking

    
    Sex=request.form["Sex"]
    if(Sex=="Erkek"):
        Sex=0
    elif(Sex=="Kadın"):
        Sex=1
    dataheart.loc[["X"], "Sex"]=Sex

    AgeCategory=request.form["AgeCategory"]
    if(AgeCategory=="18-24"):
            AgeCategory=1
    elif(AgeCategory=="25-29"):
            AgeCategory=2
    elif(AgeCategory=="30-34"):
            AgeCategory=3
    elif(AgeCategory=="35-39"):
            AgeCategory=4
    elif(AgeCategory=="40-44"):
            AgeCategory=5
    elif(AgeCategory=="45-49"):
            AgeCategory=6
    elif(AgeCategory=="50-54"):
            AgeCategory=7
    elif(AgeCategory=="55-59"):
            AgeCategory=8
    elif(AgeCategory=="60-64"):
            AgeCategory=9
    elif(AgeCategory=="65-69"):
            AgeCategory=10
    elif(AgeCategory=="70-74"):
            AgeCategory=11  
    elif(AgeCategory=="75-79"):
            AgeCategory=12
    elif(AgeCategory=="80 veya daha yaşlı"):
            AgeCategory=13  
    dataheart.loc[["X"], "AgeCategory"]=AgeCategory


    Race=request.form["Race"]
    if(Race=="Beyaz"):
            Race=1
    elif(Race=="Siyah"):
            Race=2
    elif(Race=="Asyalı"):
            Race=3
    elif(Race=="American Indian/Alaskan Native"):
            Race=4
    elif(Race=="Other"):
            Race=5
    elif(Race=="Latin"):
            Race=6
    dataheart.loc[["X"], "Race"]=Race

    Diabetic=request.form["Diabetic"]
    if(Diabetic=="Evet"):
        Diabetic=1
    elif(Diabetic=="Hayır"):
        Diabetic=0
    dataheart.loc[["X"], "Diabetic"]=Diabetic


    PhysicalActivity=request.form["PhysicalActivity"]
    if(PhysicalActivity=="Evet"):
        PhysicalActivity=0
    elif(PhysicalActivity=="Hayır"):
        PhysicalActivity=1
    dataheart.loc[["X"], "PhysicalActivity"]=PhysicalActivity



    GenHealth=request.form["GenHealth"]
    if(GenHealth=="Poor"):
        GenHealth=0
    elif(GenHealth=="Fair"):
        GenHealth=1
    elif(GenHealth=="Good"):
        GenHealth=2
    elif(GenHealth=="Very Good"):
        GenHealth=3
    elif(GenHealth=="Excellent"):
        GenHealth=4
    dataheart.loc[["X"], "GenHealth"]=GenHealth


    SleepTime=request.form["SleepTime"]
    dataheart.loc[["X"], "SleepTime"]=SleepTime



    Asthma=request.form["Asthma"]
    if(Asthma=="Evet"):
        Asthma=1
    elif(Asthma=="Hayır"):
        Asthma=0
    dataheart.loc[["X"], "Asthma"]=Asthma



    KidneyDisease=request.form["KidneyDisease"]
    if(KidneyDisease=="Evet"):
        KidneyDisease=1
    elif(KidneyDisease=="Hayır"):
        KidneyDisease=0
    dataheart.loc[["X"], "KidneyDisease"]=KidneyDisease  



    SkinCancer=request.form["SkinCancer"]
    if(SkinCancer=="Evet"):
        SkinCancer=1
    elif(SkinCancer=="Hayır"):
        SkinCancer=0
    dataheart.loc[["X"], "SkinCancer"]=SkinCancer

    print(dataheart)

    tahmin=model_heart.predict(dataheart)
    prediction_prob = model_heart.predict_proba(dataheart)
    result=''
    filenames=['']
    if tahmin==[0]:
        result=("Kalp Hastalığına Sahip olma Oranınız: %"+str(round(prediction_prob[0][1]*100,2))+"   İYİ GÖRÜNÜYORSUNUZ :)") 
        #filenames=['Bild3.png']
    elif tahmin==[1]:
        result=("Kalp Hastalığına Sahip olma Oranınız: %"+str(round(prediction_prob[0][1]*100,2))+"   KÖTÜ DURUMDASINIZ LÜTFEN DİKKATLİ OLUNUZ :)")
        #result='İYİ HUYLU NODÜL-KANSER HÜCRESİ SAPTANMAMIŞTIR.'
        #filenames=['HappyDoctorV2.jpg']

    #ths=open('C:\\Users\\Berk\\Desktop\\health\\logs.txt',"w")
    #ths.write(str(datax))
    return render_template('heartattack.html',value=result, filenames=filenames)





  ####path VERİNCE ÇALIŞIYOR...
@app.route('/brain', methods=['POST','GET'])
def brain_tumor():
    model_brain=load_model("models/model_brain/brain_model.h5")
    img=Image.open(request.files["brain"])
    new=np.array(img.resize((128,128)))
    new=new.reshape(1,128,128,3)
    newres=model_brain.predict_on_batch(new)
    newclassification=np.where(newres==np.max(newres))[1][0]
    ax=(str(newres[0][newclassification]*100) + '% Olasılıkla ')
    result=''
    if newclassification==0:
        result=ax+'TÜMÖR'
    elif newclassification==1:
        result=ax+'TÜMÖR DEĞİL'




    return render_template('braintumor.html',value=result)








if __name__=="__main__":
    app.run(debug=True)

# predictedheart=model_heart.predict(dataheart)
# result=''
#     filenames=['']
#     if predictedheart==[1]:
#         #result='KÖTÜ HUYLU KANSER HÜCRESİ-KANSER HÜCRESİ SAPTANMIŞTIR'
#         filenames=['Bild3.png']
#     elif predictedheart==[0]:
#         #result='İYİ HUYLU NODÜL-KANSER HÜCRESİ SAPTANMAMIŞTIR.'
#         filenames=['HappyDoctorV2.jpg']

#     #ths=open('C:\\Users\\Berk\\Desktop\\health\\logs.txt',"w")
#     #ths.write(str(datax))

#     return render_template('breastcancer.html',value=result, filenames=filenames)






#   PhysicalHealth=request.form.get("PhysicalHealth")
#   MentalHealth=request.form.get("MentalHealth")
#   DiffWalking=request.form.get("DiffWalking")
#   Sex=request.form.get("Sex")
#   AgeCategory=request.form.get("AgeCategory")
#   Race=request.form.get("Race")
#   Diabetic=request.form.get("Diabetic")
#   PhysicalActivity=request.form.get("PhysicalActivity")
#   GenHealth=request.form.get("GenHealth")
#   SleepTime=request.form.get("SleepTime")
#   Asthma=request.form.get("Asthma")
#   KidneyDisease=request.form.get("KidneyDisease")
#   SkinCancer=request.form.get("SkinCancer")

# if (BMI == "" or Smoking == "" or AlcoholDrinking == "" or Stroke == "" or PhysicalHealth == "" or  MentalHealth == "" or DiffWalking == "" or Sex == ""  or AgeCategory == "" or Race == "" or Diabetic ==  "" or PhysicalActivity == "" or GenHealth=="" or SleepTime=="" or Asthma=="" or KidneyDisease=="" or SkinCancer ):
#     return render_template("error.html")
# else:
    # BMI_edit(BMI)
    # Smoking_edit(Smoking)
    # AlcoholDrinking_edit(AlcoholDrinking)
    # Stroke_edit(Stroke)
    # PhysicalHealth_edit(PhysicalHealth)
    # MentalHealth_edit(MentalHealth)
    # DiffWalking_edit(DiffWalking)
    # Sex_edit(Sex)
    # AgeCategory_edit(AgeCategory)
    # Race_edit(Race)
    # Diabetic_edit(Diabetic)
    # PhysicalActivity_edit(PhysicalActivity)
    # GenHealth_edit(GenHealth)
    # SleepTime_edit(SleepTime)
    # Asthma_edit(Asthma)
    # KidneyDisease_edit(KidneyDisease)
    # SkinCancer_edit(SkinCancer)


    # yeni_veri = [[BMI_edit],[Smoking_edit],[AlcoholDrinking_edit],[Stroke_edit],[PhysicalHealth_edit],[MentalHealth_edit],[DiffWalking_edit],[Sex_edit],[AgeCategory_edit],[Race_edit],[Diabetic_edit],[PhysicalActivity_edit],[GenHealth_edit],[SleepTime_edit],[Asthma_edit],[KidneyDisease_edit],[SkinCancer_edit]]  
    # yeni_veri = pd.DataFrame(yeni_veri).T


    # df_2 = yeni_veri.rename(columns = {0:"BMI",
    #                         1:"Smoking",
    #                         2:"AlcoholDrinking",
    #                         3:"Stroke",
    #                         4:"PhysicalHealth",
    #                         5:"MentalHealth",
    #                         6:"DiffWalking",
    #                         7:"Sex",
    #                         8:"AgeCategory",
    #                         9:"Race",
    #                         10:"Diabetic",
    #                         11:"PhysicalActivity",
    #                         12:"GenHealth",
    #                         13:"SleepTime",
    #                         14:"Asthma",
    #                         15:"KidneyDisease",
    #                         16:"SkinCancer"})

    # pred=model_heart.predict(df_2)

    # return render_template("heart_hesapla.html",pred = pred, BMI = BMI, Smoking = Smoking, AlcoholDrinking = AlcoholDrinking, Stroke = Stroke, PhysicalHealth = PhysicalHealth, MentalHealth = MentalHealth, DiffWalking = DiffWalking, Sex = Sex, AgeCategory = AgeCategory, Race = Race, Diabetic = Diabetic, PhysicalActivity = PhysicalActivity,GenHealth=GenHealth,SleepTime=SleepTime,Asthma=Asthma,KidneyDisease=KidneyDisease,SkinCancer=SkinCancer)


