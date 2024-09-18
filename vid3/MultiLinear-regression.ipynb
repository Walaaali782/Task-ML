{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   area  bedrooms  age   price\n",
      "0  2600       3.0   20  550000\n",
      "1  3000       4.0   15  565000\n",
      "2  3200       NaN   18  610000\n",
      "3  3600       3.0   30  595000\n",
      "4  4000       5.0    8  760000\n",
      "5  4100       6.0    8  810000\n",
      "[  112.06244194 23388.88007794 -3231.71790863]\n",
      "221323.0018654043\n",
      "498408.2515740243\n",
      "578876.0374840144\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\MO\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n",
      "c:\\Users\\MO\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn import linear_model\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "df = pd.read_csv(\"homeprices.csv\")\n",
    "print(df)\n",
    "\n",
    "\n",
    "\n",
    "import math\n",
    "median_badrooms = math.floor(df.bedrooms.median())\n",
    "median_badrooms\n",
    "\n",
    "df.bedrooms = df.bedrooms.fillna(median_badrooms)\n",
    "df\n",
    "\n",
    "\n",
    "reg =  linear_model.LinearRegression()\n",
    "reg.fit(df[['area','bedrooms','age']],df.price)\n",
    "\n",
    "m= reg.coef_\n",
    "print(m)\n",
    "# [  112.06244194 23388.88007794 -3231.71790863]\n",
    "\n",
    "b= reg.intercept_\n",
    "print(b)\n",
    "# 221323.0018654043\n",
    "\n",
    "\n",
    "# task1 \n",
    "reg.predict([[3000,3,40]])\n",
    "# array([498408.25158031])\n",
    "\n",
    "Y =  112.06244194*3000 + 23388.88007794*3 + -3231.71790863*40 + 221323.0018654043\n",
    "print(Y) #[498408.2515740243]\n",
    "\n",
    "\n",
    "# task1 \n",
    "reg.predict([[2500,4,5]])\n",
    "# array([578876.03748933])\n",
    "\n",
    "Y =  112.06244194*2500 + 23388.88007794*4 + -3231.71790863*5 + 221323.0018654043\n",
    "print(Y) #[578876.0374840144]\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# %matplotlib inline\n",
    "# plt.xlabel('area(sqr ft)', fontsize=20)\n",
    "# plt.ylabel('price(US$)' , fontsize=20 )\n",
    "# plt.scatter(df.area,df.price,color='red',marker='+')\n",
    "\n",
    "# reg.predict([[3300]])\n",
    "# print(reg.predict([[3300]]))\n",
    "\n",
    "# m= reg.coef_\n",
    "\n",
    "# b= reg.intercept_\n",
    "\n",
    "# Y = m*3300 + b \n",
    "# print(Y) #[628715.75342466]\n",
    "\n",
    "# %matplotlib inline\n",
    "# plt.xlabel('area', fontsize=20)\n",
    "# plt.ylabel('price' , fontsize=20 )\n",
    "# plt.scatter(df.area,df.price,color='red',marker='+')\n",
    "# plt.plot(df.area,reg.predict(df[['area']]),color='blue')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
