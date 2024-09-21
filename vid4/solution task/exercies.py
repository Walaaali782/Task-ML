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
# {
#  "cells": [
#   {
#    "cell_type": "code",
#    "execution_count": 2,
#    "metadata": {},
#    "outputs": [
#     {
#      "name": "stdout",
#      "output_type": "stream",
#      "text": [
#       "    year  per capita income (US$)\n",
#       "0   1970              3399.299037\n",
#       "1   1971              3768.297935\n",
#       "2   1972              4251.175484\n",
#       "3   1973              4804.463248\n",
#       "4   1974              5576.514583\n",
#       "5   1975              5998.144346\n",
#       "6   1976              7062.131392\n",
#       "7   1977              7100.126170\n",
#       "8   1978              7247.967035\n",
#       "9   1979              7602.912681\n",
#       "10  1980              8355.968120\n",
#       "11  1981              9434.390652\n",
#       "12  1982              9619.438377\n",
#       "13  1983             10416.536590\n",
#       "14  1984             10790.328720\n",
#       "15  1985             11018.955850\n",
#       "16  1986             11482.891530\n",
#       "17  1987             12974.806620\n",
#       "18  1988             15080.283450\n",
#       "19  1989             16426.725480\n",
#       "20  1990             16838.673200\n",
#       "21  1991             17266.097690\n",
#       "22  1992             16412.083090\n",
#       "23  1993             15875.586730\n",
#       "24  1994             15755.820270\n",
#       "25  1995             16369.317250\n",
#       "26  1996             16699.826680\n",
#       "27  1997             17310.757750\n",
#       "28  1998             16622.671870\n",
#       "29  1999             17581.024140\n",
#       "30  2000             18987.382410\n",
#       "31  2001             18601.397240\n",
#       "32  2002             19232.175560\n",
#       "33  2003             22739.426280\n",
#       "34  2004             25719.147150\n",
#       "7   1977              7100.126170\n",
#       "8   1978              7247.967035\n",
#       "9   1979              7602.912681\n",
#       "10  1980              8355.968120\n",
#       "11  1981              9434.390652\n",
#       "12  1982              9619.438377\n",
#       "13  1983             10416.536590\n",
#       "14  1984             10790.328720\n",
#       "15  1985             11018.955850\n",
#       "16  1986             11482.891530\n",
#       "17  1987             12974.806620\n",
#       "18  1988             15080.283450\n",
#       "19  1989             16426.725480\n",
#       "20  1990             16838.673200\n",
#       "21  1991             17266.097690\n",
#       "22  1992             16412.083090\n",
#       "23  1993             15875.586730\n",
#       "24  1994             15755.820270\n",
#       "25  1995             16369.317250\n",
#       "26  1996             16699.826680\n",
#       "27  1997             17310.757750\n",
#       "28  1998             16622.671870\n",
#       "29  1999             17581.024140\n",
#       "30  2000             18987.382410\n",
#       "31  2001             18601.397240\n",
#       "32  2002             19232.175560\n",
#       "33  2003             22739.426280\n",
#       "34  2004             25719.147150\n",
#       "35  2005             29198.055690\n",
#       "36  2006             32738.262900\n",
#       "37  2007             36144.481220\n",
#       "38  2008             37446.486090\n",
#       "39  2009             32755.176820\n",
#       "40  2010             38420.522890\n",
#       "41  2011             42334.711210\n",
#       "42  2012             42665.255970\n",
#       "43  2013             42676.468370\n",
#       "44  2014             41039.893600\n",
#       "45  2015             35175.188980\n",
#       "46  2016             34229.193630\n",
#       "Expected per capita income in 2020: 41288.69409441762\n",
#       "[41288.69409442]\n"
#      ]
#     },
#     {
#      "name": "stderr",
#      "output_type": "stream",
#      "text": [
#       "c:\\Users\\MO\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
#       "  warnings.warn(\n"
#      ]
#     },
#     {


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





