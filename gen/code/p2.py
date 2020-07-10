# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:18:55 2020

@author: user
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

d=os.path.dirname(os.getcwd())
x=d
f=os.path.join(d,"States")
e=os.path.join(f,"Gujrat")
q=os.path.join(e,"Amreli")
w=os.path.join(q,"jan")
os.chdir(w)

dataset = pd.read_csv("jan.csv.csv")
x=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,-1].values 

l=os.path.join(x,"2020")
j=os.path.join(l,"jan")
os.chdir(j)

dataset2 = pd.read_csv("jan.csv")
x2=dataset2.iloc[:,:].values
#y2=dataset2.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,0]=labelencoder.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
labelencoder=LabelEncoder()
x[:,-1]=labelencoder.fit_transform(x[:,-1])
onehotencoder1=OneHotEncoder(categorical_features=[-1])
x=onehotencoder1.fit_transform(x).toarray()
x=x[:,1:]


labelencoder=LabelEncoder()
x2[:,0]=labelencoder.fit_transform(x2[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x2=onehotencoder.fit_transform(x2).toarray()
labelencoder=LabelEncoder()
x2[:,-1]=labelencoder.fit_transform(x2[:,-1])
onehotencoder=OneHotEncoder(categorical_features=[-1])
x2=onehotencoder.fit_transform(x2).toarray()
x2=x2[:,1:]
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)'''


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)

y_pred=regressor.predict(x2)
#plt.plot(y2,color='red',label='real')
#plt.plot(y_pred,color='blue',label='pred')
plt.title('Cotton price') 
plt.xlabel('time')
#plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
xi = list(range(len(x)))
plt.plot(xi, y_pred, marker='o', linestyle='--', color='b', label='Square') 
plt.xticks(xi, x)
plt.legend
plt.show()



