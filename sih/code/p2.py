# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:18:55 2020

@author: user
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("D:/")

d=os.path.dirname(os.getcwd())
d
xn=os.path.join(d,"sih_2/sih")
f=os.path.join(d,"sih_2/sih/States")
e=os.path.join(f,"telangana")
q=os.path.join(e,"Warangal")
os.chdir(q)

dataset = pd.read_csv("jan.csv")
x=dataset.iloc[:,:-1].values 
x=x[:-40]
x
y=dataset.iloc[:,-1].values 
y
yy=y[80:-20]
yy
y=y[:-40]
y
l=os.path.join(xn,"year")
#j=os.path.join(l,"jan") 
os.chdir(l)

dataset2 = pd.read_csv("jan.csv")
x2=dataset2.iloc[:,:-1].values
#y2=dataset2.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

label_encoder_x_1 = LabelEncoder()
x[: , 0] = label_encoder_x_1.fit_transform(x[:,0])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
x = transformer.fit_transform(x.tolist())
x = x.astype('float64')

label_encoder_x_2 = LabelEncoder()
x[: , -1] = label_encoder_x_1.fit_transform(x[:,-1])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [-1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
x = transformer.fit_transform(x.tolist())
x = x.astype('float64')
x=x[:,1:]




label_encoder_x_2 = LabelEncoder()
x2[: , 0] = label_encoder_x_1.fit_transform(x2[:,0])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
x2 = transformer.fit_transform(x2.tolist())
x2 = x2.astype('float64')

label_encoder_x_2 = LabelEncoder()
x2[: , -1] = label_encoder_x_1.fit_transform(x2[:,-1])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [-1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
x2 = transformer.fit_transform(x2.tolist())
x2 = x2.astype('float64')
x2=x2[:,1:]




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
plt.plot(xi, yy, marker='o', linestyle='--', color='r', label='Square') 
plt.xticks(xi, x)
plt.legend
plt.show()