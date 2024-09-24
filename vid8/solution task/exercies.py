from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


iris = load_iris()


print(iris.DESCR)


X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LogisticRegression(max_iter=200)  
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"Score: {accuracy}")


y_predicted = model.predict(X_test)


cm = confusion_matrix(y_test, y_predicted)
print(cm)


plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


sample_predictions = model.predict(X_test[:5])
print("Predictions for first 5 samples:", sample_predictions)

