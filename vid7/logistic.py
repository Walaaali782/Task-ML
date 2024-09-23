import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
# %matplotlib inline




df = pd.read_csv("insurance_data.csv")
df

plt.scatter(df.age,df.bought_insurance,marker='+',color='blue')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)

print(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)


y_predicted = model.predict(X_test)
y_predicted

model.predict_proba(X_test)

model.score(X_test,y_test)
y_predicted


X_test

