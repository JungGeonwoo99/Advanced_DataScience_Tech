import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import line
from sklearn import linear_model
import sklearn
import statsmodels.api as sm
from mpl_toolkits import mplot3d
from random import randint

# Problem 2
# 1. Gradient Descent Algorithm

# (STEP1) Checking for Linearity
Data = pd.read_csv("3rd_example.csv")
y=Data["y"]
z1=Data["z1"]
z2=Data["z2"]
z=Data.loc[:,["z1","z2"]]
plt.scatter(y,z1,color="orange",label="y vs z1")
plt.scatter(y,z2,color="green",label="y vs z2")
plt.legend
plt.show()

# (STEP2) Initial values : Beta, Learning rate, # of iteration
beta0=0
beta1=0
beta2=0

L = 0.0001 # The Learning rate
n = float(len(z1)) # 400

# (STEP3) Initial Iteration
Y_pred = beta0 + beta1*z1 + beta2*z2
D_beta0 = (-2/n) * sum(y-Y_pred)
D_beta1 = (-2/n) * sum(z1*(y-Y_pred))
D_beta2 = (-2/n) * sum(z2*(y-Y_pred))
beta0 = beta0 - L* D_beta0
beta1 = beta1 - L* D_beta1
beta2 = beta2 - L* D_beta2
Y_pred_new = beta0 + beta1*z1 + beta2*z2

# (STEP4) Second ~ Last Iteration
while np.sqrt(sum((Y_pred_new[i]-Y_pred[i])**2 for i in range(0,int(n)))) >= 1e-06:
# while "Distance between Y_pred_new and Y_pred" is greater than or equal to 1e-06
    Y_pred = Y_pred_new
    D_beta0 = (-2/n) * sum(y-Y_pred)
    D_beta1 = (-2/n) * sum(z1*(y-Y_pred))
    D_beta2 = (-2/n) * sum(z2*(y-Y_pred))
    beta0 = beta0 - L* D_beta0
    beta1 = beta1 - L* D_beta1
    beta2 = beta2 - L* D_beta2
    Y_pred_new = beta0 + beta1*z1 + beta2*z2
    
print(beta0, beta1, beta2)
# beta0 = 0.58686894
# beta1 = 0.41968599
# beta2 = 0.44661086

# (STEP5) Checking the estimator of Beta
model = sm.OLS(y, sm.add_constant(z))
result = model.fit()
result.summary()
'''
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.423
Model:                            OLS   Adj. R-squared:                  0.420
Method:                 Least Squares   F-statistic:                     145.4
Date:                Wed, 15 Jun 2022   Prob (F-statistic):           4.28e-48
Time:                        12:07:12   Log-Likelihood:                -661.21
No. Observations:                 400   AIC:                             1328.
Df Residuals:                     397   BIC:                             1340.
Df Model:                           2
Covariance Type:            nonrobust
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.5878      0.095      6.207      0.000       0.402       0.774
z1             0.4197      0.046      9.195      0.000       0.330       0.509
z2             0.4462      0.054      8.319      0.000       0.341       0.552
==============================================================================
Omnibus:                        3.276   Durbin-Watson:                   2.102
Prob(Omnibus):                  0.194   Jarque-Bera (JB):                3.231
Skew:                           0.179   Prob(JB):                        0.199
Kurtosis:                       2.744   Cond. No.                         4.14
==============================================================================
'''
# (STEP6) Conclusion : In OLS Regression Results, we can find "coef of [const/z1/z2]", which are estimated by [beta0/beta1/beta2]

# 2. Stochastic Gradient Descent Algorithm
# (STEP1) Initial values : Beta, Learning rate, # of iteration
beta0=0
beta1=0
beta2=0

L = 0.0001 # The Learning rate
n = float(len(z)) # 400

# (STEP2) Initial Iteration
Y_pred = beta0 + beta1*z1 + beta2*z2
D_beta0 = (-2/n) * sum(y-Y_pred)
D_beta1 = (-2/n) * sum(z1*(y-Y_pred))
D_beta2 = (-2/n) * sum(z2*(y-Y_pred))
beta0 = beta0 - L* D_beta0
beta1 = beta1 - L* D_beta1
beta2 = beta2 - L* D_beta2
Y_pred_new = beta0 + beta1*z1 + beta2*z2

# (STEP3) Second ~ Last Iteration
while np.sqrt(sum((Y_pred_new[i]-Y_pred[i])**2 for i in range(0,int(n)))) >= 1e-06:
# while "Distance between Y_pred_new and Y_pred" is greater than or equal to 1e-06
    Y_pred = Y_pred_new
    i=randint(0,n-1)
    D_beta0 = (-2) * (y-Y_pred)[i]
    D_beta1 = (-2) * z1[i]*(y-Y_pred)[i]
    D_beta2 = (-2) * z2[i]*(y-Y_pred)[i]
    beta0 = beta0 - L* D_beta0
    beta1 = beta1 - L* D_beta1
    beta2 = beta2 - L* D_beta2
    Y_pred_new = beta0 + beta1*z1 + beta2*z2
    
print(beta0, beta1, beta2)
# beta0 = 0.58686894
# beta1 = 0.41567790
# beta2 = 0.44694928

# (STEP4) Conclusion : We can estimate "[beta0/beta1/beta2]" which are also near by "coef of [const/z1/z2]"


model = sm.OLS(z2, sm.add_constant(z1))
result = model.fit()
result.summary()
"""
                            OLS Regression Results
==============================================================================
Dep. Variable:                     z2   R-squared:                       0.223
Model:                            OLS   Adj. R-squared:                  0.221
Method:                 Least Squares   F-statistic:                     114.1
Date:                Fri, 17 Jun 2022   Prob (F-statistic):           1.38e-23
Time:                        18:42:49   Log-Likelihood:                -634.60
No. Observations:                 400   AIC:                             1273.
Df Residuals:                     398   BIC:                             1281.
Df Model:                           1
Covariance Type:            nonrobust
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.0865      0.070     15.575      0.000       0.949       1.224
z1             0.4017      0.038     10.682      0.000       0.328       0.476
==============================================================================
Omnibus:                        3.725   Durbin-Watson:                   2.099
Prob(Omnibus):                  0.155   Jarque-Bera (JB):                3.600
Skew:                           0.186   Prob(JB):                        0.165
Kurtosis:                       2.720   Cond. No.                         2.40
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

model = sm.OLS(y, sm.add_constant(z1))
result = model.fit()
result.summary()
