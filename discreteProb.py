#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 22:10:41 2025

@author: Ofer Shir, oshir@alumni.Princeton.EDU
"""
import numpy as np
import matplotlib.pyplot as plt
#
def shifted_binomial_random(S, p=0.5, size=1):
    # Calculate n from sigma (since n = 4 * sigma^2)
    N = int(0.5 * np.pi * (S/1)**2)
    
    # Generate binomial random variables
    binomial_samples = np.random.binomial(N, p, size)
    
    # Shift the samples by subtracting N/2
    shifted_samples = binomial_samples - (N // 2)
    
    return shifted_samples
#
def symmetrical_uniform_randint_from_sigma(S, size=1):
    # Calculate n from sigma (since n = 4 * sigma^2)
    N = int((2*(S/1) - 1 + np.sqrt(1+(S/1)**2)) / 2)
    
    # Generate a symmetric uniform random integer vector with given
    random_vector = np.random.randint(-N, N + 1, size=size)
    
    return random_vector
#
# num_samples = 10000  # Number of samples
# #
# def targetMeasure(data) :
#     l1_norms = np.sum(np.abs(data), axis=0)    
#     empirical_expected_l1_norm = np.mean(l1_norms)
#     return empirical_expected_l1_norm
# n_values = [2] #[2, 10, 30, 80]  # List of different dimensionality values
# S1 = 1.0 
# S_values = np.linspace(1, 50, 50)
# plt.figure(figsize=(8, 6))
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
# # Loop over different values of S
# for idx, n in enumerate(n_values):
#     # title = f'sigma-vs-S_N{n}'
#     L_values = S1 + (n-1)*S_values
#     nDU,nSB = np.zeros_like(S_values),np.zeros_like(S_values)
#     for i, S2 in enumerate(S_values):
#         DU_samples = [] 
#         DU_samples.append(symmetrical_uniform_randint_from_sigma(S=S1,size=num_samples))
#         for _ in range(n-1) :
#         # Generate samples with mean = 0 and given variance (std = sqrt(variance))
#             DU_samples.append(symmetrical_uniform_randint_from_sigma(S=S2,size=num_samples))
#         DU_samples = np.vstack(DU_samples)
#         nDU[i] = targetMeasure(DU_samples)
        
#         SB_samples = [] 
#         SB_samples.append(shifted_binomial_random(S=S1, size=num_samples))
#         for _ in range(n-1) :
#         # Generate samples with mean = 0 and given variance (std = sqrt(variance))
#             SB_samples.append(shifted_binomial_random(S=S2, size=num_samples))
#         SB_samples = np.vstack(SB_samples)
#         nSB[i] = targetMeasure(SB_samples)
# #
#     plt.scatter(S_values, nDU, label=f"DU in {n}D",marker='s', edgecolor=colors[2*idx+1], facecolor='none', s=10)
#     plt.scatter(S_values, nSB, label=f"SB in {n}D",marker='*', color=colors[2*idx], s=10)
#     plt.plot(S_values, L_values, color='black', linestyle='--')
#     plt.yscale('log')