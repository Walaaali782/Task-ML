{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\MO\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n",
      "c:\\Users\\MO\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([681241.6684584])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "\n",
    "\n",
    "df = pd.read_csv(\"homeprices.csv\")\n",
    "df\n",
    "\n",
    "dummies = pd.get_dummies(df['town'])\n",
    "dummies\n",
    "\n",
    "\n",
    "merged = pd.concat([df,dummies],axis='columns')\n",
    "merged\n",
    "\n",
    "final = merged.drop(['town' ,'west windsor'], axis='columns')\n",
    "final\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "model = LinearRegression()\n",
    "\n",
    "\n",
    "\n",
    "X = final.drop('price', axis='columns')\n",
    "X\n",
    "\n",
    "y = final.price\n",
    "y\n",
    "\n",
    "model.fit(X,y)\n",
    "\n",
    "\n",
    "model.predict([[2800,0,1]])\n",
    "\n",
    "model.predict([[3400,0,0]])\n",
    "\n",
    "model.score(X,y)\n",
    "\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "le = LabelEncoder()\n",
    "\n",
    "dfle = df\n",
    "dfle.town = le.fit_transform(dfle.town)\n",
    "dfle\n",
    "\n",
    "\n",
    "X = dfle[['town','area']].values\n",
    "X\n",
    "\n",
    "y = dfle.price\n",
    "y\n",
    "\n",
    "\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')\n",
    "\n",
    "X = ct.fit_transform(X)\n",
    "X\n",
    "\n",
    "X = X[:,1:]\n",
    "X\n",
    "\n",
    "\n",
    "model.fit(X,y)\n",
    "\n",
    "model.predict([[0,1,3400]]) # 3400 sqr ft home in west windsor\n",
    "\n",
    "model.predict([[1,0,2800]])\n"
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
