# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:41:32 2024

@author: Ofer Shir, oshir@alumni.Princeton.EDU
"""
import numpy as np

def angles_to_covariance(rot_matrix):
    """
    Transforms a rotation matrix to a covariance matrix.

    Args:
        rot_matrix: The input rotation matrix.

    Returns:
        The corresponding covariance matrix.
    """

    n = len(rot_matrix)
    cov_matrix = np.eye(n)

    for k in range(n):
        for l in range(k, n):
            if k == l:
                cov_matrix[k, k] = rot_matrix[k, k]**2
            else:
                cov_matrix[k, l] = 0.5 * (rot_matrix[k, k]**2 - rot_matrix[l, l]**2) * np.tan(2 * rot_matrix[k, l])
                cov_matrix[l, k] = cov_matrix[k, l]

    # Ensure positive semi-definiteness
    cov_matrix = enforce_symmetry_and_psd(cov_matrix)
    return cov_matrix

def enforce_symmetry_and_psd(matrix):
    # Ensure symmetry
    matrix = (matrix + matrix.T) / 2

    # Ensure positive semidefinite using eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues[eigenvalues < 0] = 0
    matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return matrix