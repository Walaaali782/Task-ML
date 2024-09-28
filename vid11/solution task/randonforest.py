import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
dir(iris)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()


df['target'] = iris.target
df.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( df.drop('target',axis='columns'),iris.target,test_size=0.2)

len(X_test)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

model.score(X_test,y_test)


model = RandomForestClassifier(n_estimators=40)
model.fit(X_train, y_train)
model.score(X_test,y_test)