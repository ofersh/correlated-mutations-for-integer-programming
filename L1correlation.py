#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:52:04 2025

@author: Ofer Shir, oshir@alumni.Princeton.EDU
"""
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def sophisticatedL1correlation(data,label) :
    X = data[:,0]
    Y = data[:,1]
    scaler = StandardScaler()
    X_scaled = X.reshape(-1, 1)  # Ensure X is a 2D array
    Y_scaled = Y.reshape(-1, 1)  # Ensure Y is a 2D array
    X_scaled = scaler.fit_transform(X_scaled)
    Y_scaled = scaler.fit_transform(Y_scaled)
    
    # Fit a Lasso regression model to predict Y from X
    lasso = Lasso(alpha=0.1)  # L1 regularization strength
    
    
    lasso.fit(X_scaled, Y_scaled)
    
    # The L1-norm of the coefficients gives us an idea of the relationship strength
    l1_norm_coefs = np.sum(np.abs(lasso.coef_))  # sum of the absolute values of the coefficients
    
    # If the L1-norm of the coefficients is greater, it indicates a stronger relationship
    print(label + f"L1-norm of the coefficients: {l1_norm_coefs}")
    
    # Now we compute the R^2 value (traditional regression correlation-like measure)
    model = sm.OLS(Y_scaled, sm.add_constant(X_scaled)).fit()
    print(label + f"R-squared (OLS): {model.rsquared}")
    
    # We can compute the L1 correlation-like measure
    # A larger L1-norm of the coefficients suggests stronger predictability
    l1_correlation_like = l1_norm_coefs / (np.mean(np.abs(X_scaled)) * np.mean(np.abs(Y_scaled)))
    print(label + f"L1-based correlation-like measure: {l1_correlation_like}")
    