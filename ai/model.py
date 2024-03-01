import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

def ai_eğitim_sonuc():
    data = pd.read_csv("/home/osman/Documents/projeler/Android-App-Malicious-anlysis/datasets/drebin-215-dataset-5560malware-9476-benign.csv")
    data = pd.DataFrame(data)
    data = data.drop('TelephonyManager.getSimCountryIso', axis=1)
    y = data["class"]
    x = data.drop("class", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    Tümsonuclar= {}
    
    modelLR = LogisticRegression()
    modelLR.fit(X_train, y_train)
    y_pred = modelLR.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    Tümsonuclar["modelLR"] = accuracy
    

    modelSVM = SVC()
    modelSVM.fit(X_train, y_train)
    y_pred = modelSVM.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    Tümsonuclar["modelSVM"] = accuracy
    
    
    modelDT = DecisionTreeClassifier()
    modelDT.fit(X_train, y_train)
    y_pred = modelDT.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    Tümsonuclar["modelDT"] = accuracy
    
    
    modelRF = RandomForestClassifier()
    modelRF.fit(X_train, y_train)
    y_pred = modelRF.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    Tümsonuclar["modelRF"] = accuracy
    
    
    

    modelMLP = MLPClassifier()
    modelMLP.fit(X_train, y_train)
    y_pred = modelMLP.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    Tümsonuclar["modelMLP"] = accuracy
    
    

    modelGNB = GaussianNB()
    modelGNB.fit(X_train, y_train)
    y_pred = modelGNB.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    Tümsonuclar["modelGNB"] = accuracy
    
    
    return Tümsonuclar