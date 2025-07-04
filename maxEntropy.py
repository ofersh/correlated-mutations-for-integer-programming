# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:29:27 2024

@author: Ofer Shir, oshir@alumni.Princeton.EDU
"""
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

# Function to compute the difference of erf terms
def exact_pk_TGauss(k, sigma):
    x1 = (k + 0.5) / (np.sqrt(2) * sigma)
    x2 = (k - 0.5) / (np.sqrt(2) * sigma)
    return 0.5*(erf(x1) - erf(x2))

def approx_pk_TGauss(k, sigma):
    c = k / (np.sqrt(2) * sigma)
    return np.exp(-c**2) / np.sqrt(np.pi)
    
def pk_dgeometric(k, p):
    normalization_factor = p / (2 - p)
    return normalization_factor * (1 - p) ** abs(k)
#
def entropy_sym_randint(sigma) :
    N = (-1 + np.sqrt(1 + 12 * sigma**2)) / 2
    return np.log2(2 * N + 1)
#
def entropy_sym_randint_from_l1norm(S) :
    N = int((2 * S - 1 + np.sqrt(1 + 4 * S**2)) / 2)
    return np.log2(2 * N + 1)
#
def entropy_dgeometric(p, k_max=1000):
    """
    Compute the entropy of the double-geometric distribution for a given p.
    
    Args:
        p (float): The parameter of the double-geometric distribution (0 < p < 1).
        k_max (int): Range of k values to approximate the infinite sum.
    
    Returns:
        float: Entropy value.
    """
    # Range of k values
    k_values = np.arange(-k_max, k_max + 1)
    
    # Compute the probabilities
    probs = np.array([pk_dgeometric(k, p) for k in k_values])
    
    # Avoid log(0) issues by filtering zero probabilities
    probs_nonzero = probs[probs > 0]
    
    # Compute entropy: H = -sum(p_k * log2(p_k))
    entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    return entropy

def entropy_TGauss(sigma, k_max=1000, exact=True):
    """
    Compute the entropy of the Truncated Gaussian distribution for a given sigma.
    
    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        k_max (int): Range of k values to approximate the infinite sum.
    
    Returns:
        float: Entropy value.
    """
    # Range of k values
    k_values = np.arange(-k_max, k_max + 1)
    
    # Compute probabilities
    if exact :
        probs = np.array([exact_pk_TGauss(k, sigma) for k in k_values])
    else :
        probs = np.array([approx_pk_TGauss(k, sigma) for k in k_values])
    
    # Normalize to ensure valid probability distribution
    probs /= np.sum(probs)
    
    # Filter out zero probabilities to avoid log(0)
    probs_nonzero = probs[probs > 0]
    
    # Compute entropy: H = -sum(p_k * log2(p_k))
    entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    return entropy


# Range of p values
p_values = np.linspace(0.01, 0.99, 100)
# Compute entropy for each p
DGentropy_values = [entropy_dgeometric(p) for p in p_values]
S_values = 2.0*(1.0-p_values)/(p_values*(2-p_values))
# Range of sigma values
sigma_values = S_values #np.linspace(0.1, 100, 500)  # From narrow to wide distributions
# Compute entropy for each sigma
TGentropy_values_exact = [entropy_TGauss(sigma) for sigma in sigma_values]
TGentropy_values_approx = [entropy_TGauss(sigma,exact=False) for sigma in sigma_values]


# Plot the results
# plt.figure(figsize=(8, 6))
# plt.plot(S_values, DGentropy_values, label='Entropy of Double-Geometric Distribution')
# plt.rc('text',usetex=True)
# plt.rc('font',family='serif')
# plt.xlabel(r"$S$",fontsize=20)
# plt.ylabel(r"Entropy",fontsize=20)
# # plt.title('Entropy of Double-Geometric Distribution vs. $p$')
# plt.grid(True)
# plt.ylim(0,10)
# # plt.legend()
# plt.show()

plt.figure(figsize=(9, 6))
plt.plot(S_values, DGentropy_values, label='Double-Geometric Distribution',linestyle='--')
plt.plot(sigma_values, TGentropy_values_exact, label='Truncated Normal using exact probability',linestyle='-')
plt.plot(sigma_values, TGentropy_values_approx, label='Truncated Normal using approximated probability',linestyle=':')
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.xlabel(r"Step-Size ($S$ or $\sigma$)",fontsize=20) # plt.xlabel(r"$\sigma$",fontsize=20)
plt.ylabel(r"Entropy",fontsize=20)
# plt.title('Entropy of Truncated Gaussian Distribution vs. $\\sigma$')
plt.grid(True)
plt.ylim(0,10)
plt.legend()
plt.show()