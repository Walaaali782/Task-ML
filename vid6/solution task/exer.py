import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("carprices.csv")
print(df)

dummies = pd.get_dummies(df['Car Model'])
dummies


merged = pd.concat([df,dummies],axis='columns')
merged

final = merged.drop(['Car Model' ,'Mercedez Benz C class'], axis='columns')
final

from sklearn.linear_model import LinearRegression
model = LinearRegression()



X = final.drop('Sell Price($)', axis='columns')
print(X)


y = final['Sell Price($)']
y

model.fit(X,y)


model.predict([[45000,4,0,0]])
# array([36991.31721061])

model.predict([[86000,7,0,1]])
# array([11080.74313219])

model.score(X,y)
# 0.9417050937281082



plt.figure(figsize=(10, 6))

plt.scatter(df['Mileage'], df['Sell Price($)'], color='black', label='Actual Data')


predicted_prices = model.predict(X)
plt.plot(df['Mileage'], predicted_prices, color='purple', linewidth=2, label='Regression Line')


plt.xlabel('Mileage')
plt.ylabel('Sell Price ($)')
plt.title('Mileage vs. Sell Price')
plt.legend()
plt.grid(True)


plt.show()