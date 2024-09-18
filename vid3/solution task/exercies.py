{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  experience  test_score(out of 10)  interview_score(out of 10)  salary($)\n",
      "0        NaN                    8.0                           9      50000\n",
      "1        NaN                    8.0                           6      45000\n",
      "2       five                    6.0                           7      60000\n",
      "3        two                   10.0                          10      65000\n",
      "4      seven                    9.0                           6      70000\n",
      "5      three                    7.0                          10      62000\n",
      "6        ten                    NaN                           7      72000\n",
      "7     eleven                    7.0                           8      80000\n",
      "[2922.26901502 2221.30909959 2147.48256637]\n",
      "14992.65144669314\n"
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
    },
    {
     "data": {
      "text/plain": [
       "array([93747.79628651])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn import linear_model\n",
    "from word2number import w2n\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "df = pd.read_csv(\"hiring.csv\")\n",
    "print(df)\n",
    "\n",
    "df.experience = df.experience.fillna(\"zero\")\n",
    "df\n",
    "\n",
    "\n",
    "df.experience = df.experience.apply(w2n.word_to_num)\n",
    "df\n",
    "\n",
    "\n",
    "import math\n",
    "median_tests= math.floor(df['test_score(out of 10)'].mean())\n",
    "median_tests\n",
    "\n",
    "df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_tests)\n",
    "df\n",
    "\n",
    "\n",
    "\n",
    "reg =  linear_model.LinearRegression()\n",
    "reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])\n",
    "\n",
    "m= reg.coef_\n",
    "print(m)\n",
    "# [2922.26901502 2221.30909959 2147.48256637]\n",
    "\n",
    "b= reg.intercept_\n",
    "print(b)\n",
    "#14992.65144669314\n",
    "\n",
    "\n",
    "# task1 \n",
    "reg.predict([[2,9,6]])\n",
    "# array([53713.86677124])\n",
    "\n",
    "# task2\n",
    "reg.predict([[12,10,10]])\n",
    "# import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn import linear_model\n",
    "from word2number import w2n\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "df = pd.read_csv(\"hiring.csv\")\n",
    "print(df)\n",
    "\n",
    "df.experience = df.experience.fillna(\"zero\")\n",
    "df\n",
    "\n",
    "\n",
    "df.experience = df.experience.apply(w2n.word_to_num)\n",
    "df\n",
    "\n",
    "\n",
    "import math\n",
    "median_tests= math.floor(df['test_score(out of 10)'].mean())\n",
    "median_tests\n",
    "\n",
    "df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_tests)\n",
    "df\n",
    "\n",
    "\n",
    "\n",
    "reg =  linear_model.LinearRegression()\n",
    "reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])\n",
    "\n",
    "m= reg.coef_\n",
    "print(m)\n",
    "# [2922.26901502 2221.30909959 2147.48256637]\n",
    "\n",
    "b= reg.intercept_\n",
    "print(b)\n",
    "#14992.65144669314\n",
    "\n",
    "\n",
    "# task1 \n",
    "reg.predict([[2,9,6]])\n",
    "# array([53713.86677124])\n",
    "\n",
    "# task2\n",
    "reg.predict([[12,10,10]])\n",
    "# import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn import linear_model\n",
    "from word2number import w2n\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "df = pd.read_csv(\"hiring.csv\")\n",
    "print(df)\n",
    "\n",
    "df.experience = df.experience.fillna(\"zero\")\n",
    "df\n",
    "\n",
    "\n",
    "df.experience = df.experience.apply(w2n.word_to_num)\n",
    "df\n",
    "\n",
    "\n",
    "import math\n",
    "median_tests= math.floor(df['test_score(out of 10)'].mean())\n",
    "median_tests\n",
    "\n",
    "df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_tests)\n",
    "df\n",
    "\n",
    "\n",
    "\n",
    "reg =  linear_model.LinearRegression()\n",
    "reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])\n",
    "\n",
    "m= reg.coef_\n",
    "print(m)\n",
    "# [2922.26901502 2221.30909959 2147.48256637]\n",
    "\n",
    "b= reg.intercept_\n",
    "print(b)\n",
    "#14992.65144669314\n",
    "\n",
    "\n",
    "# task1 \n",
    "reg.predict([[2,9,6]])\n",
    "# array([53713.86677124])\n",
    "\n",
    "# task2\n",
    "reg.predict([[12,10,10]])\n",
    "# array([93747.79628651])\n",
    "\n"
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
