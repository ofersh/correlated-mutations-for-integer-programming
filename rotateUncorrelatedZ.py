# -*- coding: utf-8 -*-
"""
Implementing the Correlated Mutation pseudo-code from the Lecture's slides.

@author: Ofer Shir, oshir@alumni.Princeton.EDU
"""
import numpy as np
from scipy.stats import uniform
from scipy.stats import norm

def rotateUncorrelatedZ(zu,alpha) :
    n = len(zu)
    zc = zu.copy()
    nq = int(n*(n-1)/2)-1
    for k in range(1,n) :
        n1 = n-k-1
        n2 = n-1
        for _ in range(k) :
            d1 = zc[n1]
            d2 = zc[n2]
            #introduce rounding here? + is the minus sign correct?
            zc[n1] = d1*np.cos(alpha[nq])- d2*np.sin(alpha[nq]) 
            zc[n2] = d1*np.sin(alpha[nq])+ d2*np.cos(alpha[nq])
            n2 -= 1
            nq -= 1
    return zc
#
def l1_correlation(X, Y) :
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    X_dev = np.mean(np.abs(X-X_mean))
    Y_dev = np.mean(np.abs(Y-Y_mean))
    if X_dev==0 or Y_dev==0 :
        return 0
    covariance_like = np.mean(np.abs((X-X_mean)*(Y-Y_mean)))
    return (covariance_like/(X_dev*Y_dev))
    #
def generatePopulation_iDG(p1=0.5, p2=0.5, alpha12=0.0, num_samples = 10000) :
    U11 = uniform.rvs(size=num_samples)
    U12 = uniform.rvs(size=num_samples)
    Z11 = np.floor(np.log(1 - U11) / np.log(1 - p1))
    Z12 = np.floor(np.log(1 - U12) / np.log(1 - p2))
    ZU1 = np.vstack((Z11,Z12)).T
    U21 = uniform.rvs(size=num_samples)
    U22 = uniform.rvs(size=num_samples)
    Z21 = np.floor(np.log(1 - U21) / np.log(1 - p1))
    Z22 = np.floor(np.log(1 - U22) / np.log(1 - p2))
    ZU2 = np.vstack((Z21,Z22)).T
#
    if alpha12 == 0.0 :
        return (ZU1-ZU2) #ZC1,ZC2 = ZU1,ZU2
    else :
        ZC1,ZC2 = np.zeros_like(ZU1),np.zeros_like(ZU2)
        for k in range(num_samples) :
            ZC1[k, :] = rotateUncorrelatedZ(ZU1[k, :], alpha12)  
            ZC2[k, :] = rotateUncorrelatedZ(ZU2[k, :], alpha12)
        return np.round(ZC1-ZC2)
#
def generatePopulation_2D_DG(p1=0.5, p2=0.5, alpha12=0.0, num_samples = 10000) :
    # Z11 = geom.rvs(p1, size=num_samples).astype(int)
    # Z12 = geom.rvs(p2, size=num_samples).astype(int)
    U11 = uniform.rvs(size=num_samples)
    U12 = uniform.rvs(size=num_samples)
    Z11 = np.floor(np.log(1 - U11) / np.log(1 - p1))
    Z12 = np.floor(np.log(1 - U12) / np.log(1 - p2))
    ZU1 = np.vstack((Z11,Z12)).T
    # Z21 = geom.rvs(p1, size=num_samples).astype(int)
    # Z22 = geom.rvs(p2, size=num_samples).astype(int)
    U21 = uniform.rvs(size=num_samples)
    U22 = uniform.rvs(size=num_samples)
    Z21 = np.floor(np.log(1 - U21) / np.log(1 - p1))
    Z22 = np.floor(np.log(1 - U22) / np.log(1 - p2))
    ZU2 = np.vstack((Z21,Z22)).T
#
    if alpha12 == 0.0 :
        return (ZU1-ZU2) #ZC1,ZC2 = ZU1,ZU2
    else :
        ZC1,ZC2 = np.zeros_like(ZU1),np.zeros_like(ZU2)
        for k in range(num_samples) :
            ZC1[k, :] = rotateUncorrelatedZ(ZU1[k, :], alpha12)  
            ZC2[k, :] = rotateUncorrelatedZ(ZU2[k, :], alpha12)
        return (np.round(ZC1) - np.round(ZC2))
#
def generatePopulation_2D_TN(sigma1=1.0, sigma2=1.0, alpha12=0.0, num_samples = 10000) :
    Z1 = norm.rvs(loc=0, scale=sigma1, size=num_samples)
    Z2 = norm.rvs(loc=0, scale=sigma2, size=num_samples)
    ZU = np.vstack((Z1,Z2)).T
    if alpha12 == 0.0 :
        ZC = ZU
    else :
        ZC = np.zeros_like(ZU)
        for k in range(num_samples) :
            ZC[k, :] = rotateUncorrelatedZ(ZU[k, :], alpha12)  
    return (np.round(ZC))
#
def analyzeSampledPopulation2D(data, label, ABSOLUTE=False) :
    l1_norm_z1 = np.mean(np.abs(data[:, 0]))  # L1 norm for Z1
    l1_norm_z2 = np.mean(np.abs(data[:, 1]))  # L1 norm for Z1
    l1_norms = np.sum(np.abs(data), axis=1)    
    empirical_expected_l1_norm = np.mean(l1_norms)

    # Print the results
    print(f"{label} Individual L1 norm for Z1 (sum of absolute values): {l1_norm_z1}")
    print(f"{label} Individual L1 norm for Z2 (sum of absolute values): {l1_norm_z2}")
    print(f"{label} Empirical aggregated expected L1 norm: {empirical_expected_l1_norm}")
    # correlation_coefficient = np.corrcoef(data[:, 0], data[:, 1])#[0, 1]
    # l1_coefficient = l1_correlation(data[:, 0], data[:, 1])
    # #spearman_correlation_coefficient = spearmanr(TN_samples[:, 0], TN_samples[:, 1])[0]
    # print(f"Pearson Correlation: {correlation_coefficient[0,1]}")
    # print(f"L1 Correlation: {l1_coefficient}")
    if ABSOLUTE :
        empirical_cov = np.cov(np.abs(data), rowvar=False)
        print("\nEmpirical " + label + " cov(ABS):")
    else :
        empirical_cov = np.cov(data, rowvar=False)
        print("\nEmpirical " + label + " covariance:")
    print(empirical_cov)
    ratio = empirical_cov[1,1]/empirical_cov[0,0]
    print(f"ratio = {ratio} ({np.sqrt(ratio)})")
    