import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

def predict_sklean():
    df = pd.read_csv("test_scores.csv")
    r = linear_model.LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.0002

    costend = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, costend, rel_tol=1e-20):
            break
        costend = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,cost, i))

    return m_curr, b_curr

if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print("gradient descent : Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_sklean()
    print("sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))







# df = pd.read_csv("test_scores.csv")
# print(df)


# x = df[['math']] 
# y = df['cs']  
# x
# y
# %matplotlib inline
# plt.xlabel('year', fontsize=20)
# plt.ylabel('per capita income (US$)' , fontsize=20 )
# plt.scatter(x,y,color='purple',marker='.')

# reg = linear_model.LinearRegression()


# reg.fit(x, y)

# year_2020 = np.array([[2020]])
# predicted_income = reg.predict(year_2020)

# print(f"Expected per capita income in 2020: {predicted_income[0]}")


# m= reg.coef_

# b= reg.intercept_

# Y = m*2020 + b 
# print(Y)


# plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+')
# plt.plot(df['year'], reg.predict(X), color='blue')
# plt.xlabel('Year')
# plt.ylabel('Per Capita Income (US$)')
# plt.title('Per Capita Income Prediction using Linear Regression')
# plt.show()





