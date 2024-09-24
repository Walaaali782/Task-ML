import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt





df = pd.read_csv("titanic.csv")
df.head()

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()

inputs = df.drop('Survived',axis='columns')
target = df.Survived


inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})


inputs.Age = inputs.Age.fillna(inputs.Age.mean())

inputs

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)

len(X_train)
len(X_test)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)





