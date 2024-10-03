from sklearn import datasets
wine = datasets.load_wine()


dir(wine)

wine.data[0:2]

wine.feature_names

wine.target_names

wine.target[0:2]

import pandas as pd
df = pd.DataFrame(wine.data,columns=wine.feature_names)
df.head()

df['target'] = wine.target
df[50:70]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=100)

from sklearn.naive_bayes import GaussianNB, MultinomialNB
model = GaussianNB()
model.fit(X_train,y_train)

model.score(X_test,y_test)


mn = MultinomialNB()
mn.fit(X_train,y_train)
mn.score(X_test,y_test)