# -*- coding: utf-8 -*-

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston

def ftest(X,y):
    n, p = X.shape
    X_data = np.concatenate((np.ones(n).reshape(-1,1), X), axis=1)
    X_squared = np.matmul(np.transpose(X_data), X_data)
    beta = np.matmul(np.matmul(np.linalg.inv(X_squared),np.transpose(X_data)), y.reshape(-1,1))
    y_pred = np.matmul(X_data, beta)
    
    SSE = sum((y.reshape(-1,1)-y_pred)**2)
    SSR = sum((y_pred-np.mean(y_pred))**2)
    SST = SSE+SSR
    
    MSE = SSE/(n-p-1)
    MSR = SSR/p
    F = MSR/MSE
    P = 1-stats.f.cdf(F,p,n-p-1)
    
    print("========================================================================================")
    print("Factor          SS                DF            MS                 F-value            pr>F")
    print("Model", "    ",SSR, "     ", p, "     ", MSR, "     ", F, "   ", P) 
    print("Error", "    ",SSE, "     ", n - p - 1, "     ", MSE)
    print("========================================================================================")
    print("Total", "    ",SSE + SSR, "     ", p + n - p - 1)
    
    return None

def ttest(X,y,varname=None):
    name = np.append('const',data.feature_names)
    n,p = X.shape
    X_data = np.concatenate((np.ones(n).reshape(-1,1), X), axis=1)
    X_squared = np.matmul(np.transpose(X_data), X_data)
    beta = np.matmul(np.matmul(np.linalg.inv(X_squared),np.transpose(X_data)), y.reshape(-1,1))
    y_pred = np.matmul(X_data, beta)
    
    SSE = sum((y.reshape(-1,1)-y_pred)**2)
    SSR = sum((y_pred-np.mean(y_pred))**2)
    SST = SSE+SSR
    
    MSE = SSE/(n-p-1)
    MSR = SSR/p
    
    se_matrix = MSE*(np.linalg.inv(np.matmul(np.transpose(X_data),X_data)))
    se_matrix = np.diag(se_matrix)
    se = np.sqrt(se_matrix)

    T = []
    for i in range(len(se_matrix)):
        T.append((beta[i] / np.sqrt(se_matrix[i])))

    P = ((1 - stats.t.cdf(np.abs(np.array(T)), n - p - 1)) * 2)

    print("==================================================================================================")
    print("Variable            coef                    se                          t               Pr>|t|")
    for i in range(0,14):
        print(name[i], "         ", beta[i], "       ", se[i], "       ", T[i], "       ", P[i])
    print("==================================================================================================")
    
    return None

## Do not change!
# load data
data=load_boston()
X=data.data
y=data.target

ftest(X,y)
ttest(X,y,varname=data.feature_names)
