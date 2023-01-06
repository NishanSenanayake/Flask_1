# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:38:21 2023

@author: 14198
"""
import pandas as pd
import seaborn as sns
import numpy as np


df=pd.read_csv("data/heart_data.csv")

df=df.drop(columns=df.columns[0],axis=1)


sns.lmplot(x='biking', y='heart.disease', data=df)
sns.lmplot(x='smoking', y='heart.disease', data=df)


x_df= df.drop('heart.disease', axis=1)

y_df=df['heart.disease']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

from sklearn import linear_model

model=linear_model.LinearRegression()

model.fit(X_train,y_train)

print(model.score(X_train,y_train))


prediction_test=model.predict(X_test)

print("mean sq. error =", np.mean(prediction_test-y_test)**2)


import pickle


pickle.dump(model,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[70.1,26.3]]))
