# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:17:56 2019

@author: Pranay Rawat
"""
import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)
print(pandas.__version__)
print(matplotlib.__version__)
print(seaborn.__version__)
print(sklearn.__version__)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

games= pandas.read_csv("games.csv")

#print the names of the columns in games
print(games.columns)
print(games.shape)


#make a histogram of all the ratings in the ratings in the average_rating column
plt.hist(games["average_rating"])
plt.show()

#print the first row of all the games with zero scores
print(games[games["average_rating"]==0].iloc[0])

#print the first row of games with score greater than 0
print(games[games["average_rating"]>0].iloc[0])


#remove any rows without user reviews
games = games[games["users_rated"]>0]

#remove any rows with missing values
games = games.dropna(axis=0)


#make histogram of all the average ratings
plt.hist(games["average_rating"])
plt.show()

#correlation matrix
corrmat = games.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()

#get all the columns from the dataframe
columns = games.columns.tolist()

#filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]

#store the variable we'll be predicting on
target = "average_rating"


#generating training and test datasets
from sklearn.model_selection import train_test_split
#generate training set
train = games.sample(frac=0.8,random_state=1)

#select anything not in the training set and put it in test
test= games.loc[~games.index.isin(train.index)]

#print shapes
print(train.shape)
print(test.shape)

#import Linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


#initialize the model class
LR=LinearRegression()

#fit the model the training data
LR.fit(train[columns],train[target])

#generate predictons for the test set
predictions = LR.predict(test[columns])

#compute error between our test predictions and actual values
mean_squared_error(predictions,test[target])

#import the random forest model
from sklearn.ensemble import RandomForestRegressor

#initialize the model
RFR= RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)

#fir to the data
RFR.fit(train[columns],train[target])

#make predictions 
predictions = RFR.predict(test[columns])

#compute the error between out test predictions
mean_squared_error(predictions,test[target])

test[columns].iloc[0]

rating_LR=LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR=RFR.predict(test[columns].iloc[0].values.reshape(1,-1))


print(rating_LR)
print(rating_RFR)
test[target].iloc[0]

