#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 13:16:52 2025

@author: Ofer Shir, oshir@alumni.Princeton.EDU
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import comb
import math

# Exact entropy
def shifted_binomial_entropy_theoretical (sigma) :
    n = int(4 * sigma**2)
    if n > 1000 :
        variance = n/4 #normal distribution approximation for large n
        entropy = 0.5 * math.log2(2*math.pi * math.e * variance)
    else :
        entropy = 0.0
        for k in range(-n//2, n//2 +1):
            try :
                probability = comb(n, k + n//2) * (0.5**n)
                if probability > 0 :
                    entropy -= probability * math.log2(probability)
            except ValueError :
                print('Edge case: comb became invalid!')
                pass
        # print(f"(n={n}) entropy={entropy}")
    return entropy

# Function to calculate the theoretical entropy of the shifted binomial distribution
def shifted_binomial_entropy_PMF(sigma, p=0.5):
    # Calculate n from sigma (since n = 4 * sigma^2)
    n = int(4 * sigma**2)
    
    # Generate the PMF for the binomial distribution (for X)
    k_values = np.arange(0, n+1)
    pmf_values = binom.pmf(k_values, n, p)
    
    # Shift the values of k by subtracting n/2
    # shifted_k_values = k_values - (n // 2)
    
    # Calculate the entropy of the shifted binomial distribution
    entropy = -np.sum(pmf_values * np.log2(pmf_values + 1e-10))  # Avoid log(0)
    
    return entropy

# Function to calculate the empirical entropy of the shifted binomial distribution
def shifted_binomial_entropy_empirical(sigma, p=0.5, num_samples=10000):
    # Calculate n from sigma (since n = 4 * sigma^2)
    n = int(4 * sigma**2)
    
    # Generate num_samples binomial random variables
    binomial_samples = np.random.binomial(n, p, num_samples)
    
    # Shift the samples by subtracting n/2
    shifted_samples = binomial_samples - (n // 2)
    
    # Estimate the PMF (probability distribution) from the samples
    unique_values, counts = np.unique(shifted_samples, return_counts=True)
    probabilities = counts / num_samples
    
    # Calculate the empirical entropy
    empirical_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Avoid log(0)
    
    return empirical_entropy

# Range of sigma values to evaluate
sigma_range = np.linspace(0.1, 25, 100)

# Compute the theoretical and empirical entropy for each sigma
theoretical_entropy_values = [shifted_binomial_entropy_theoretical(sigma) for sigma in sigma_range]
pmf_entropy_values = [shifted_binomial_entropy_PMF(sigma) for sigma in sigma_range]
empirical_entropy_values = [shifted_binomial_entropy_empirical(sigma) for sigma in sigma_range]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(sigma_range, theoretical_entropy_values, label='Theoretical Entropy', c='blue')
plt.plot(sigma_range, pmf_entropy_values, label='PMF Entropy', color='green')
plt.plot(sigma_range, empirical_entropy_values, label='Empirical Entropy', color='red', linestyle='--')
plt.title('Entropy Calculations of Shifted Binomial Distribution')
plt.xlabel('$\\sigma$ (Standard Deviation)')
plt.ylabel('Entropy (bits)')
plt.legend()
plt.grid(True)
plt.show()
