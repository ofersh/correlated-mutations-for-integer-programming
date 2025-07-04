# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:02:52 2024

@author: Ofer Shir, oshir@alumni.Princeton.EDU
"""
# from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import entropy
import scipy.stats as stats
from numpy.linalg import LinAlgError
from scipy.spatial import cKDTree
from scipy.special import gamma
from scipy.stats import kde
#
def joint_entropy_2d_histogram(samples, method='auto'):
    """
    Estimates the entropy of a 2D distribution using a histogram.

    Args:
        samples: A NumPy array of shape (num_samples, 2) containing the samples.
        method: Method for determining bin width:
            - 'scott' (Scott's rule)
            - 'freedman' (Freedman-Diaconis rule)
            - 'auto' (selects between 'scott' and 'freedman' based on data)

    Returns:
        An estimate of the differential entropy of the distribution.

    Raises:
        ValueError: If bin width calculation fails or results in invalid values.
    """

    try:
        bins_x, bins_y = _determine_bins(samples, method)

        hist_tuple = np.histogram2d(samples[:, 0], samples[:, 1], bins=(bins_x, bins_y), density=True)
        hist, _, _ = hist_tuple
        hist = hist.astype(float)  # Ensure floating-point for accurate calculations
        hist /= hist.sum()        # Normalize to obtain probability distribution

      # Handle potential zero probabilities (log(0) is undefined)
        hist[hist == 0] = 1e-12  # Small value to avoid log(0)

        joint_entropy = -np.sum(hist * np.log2(hist))

        return joint_entropy

    except (ValueError, ZeroDivisionError) as e:
        print(f"Error in entropy estimation: {e}")
        return None  # Or handle the error as needed (e.g., return a default value)
    #
def estimate_entropy_2d_histogram(samples, method='auto'):
    """
    Estimates the entropy of a 2D distribution using a histogram.

    Args:
        samples: A NumPy array of shape (num_samples, 2) containing the samples.
        method: Method for determining bin width:
            - 'scott' (Scott's rule)
            - 'freedman' (Freedman-Diaconis rule)
            - 'auto' (selects between 'scott' and 'freedman' based on data)

    Returns:
        An estimate of the differential entropy of the distribution.

    Raises:
        ValueError: If bin width calculation fails or results in invalid values.
    """

    try:
        bins_x, bins_y = _determine_bins(samples, method)

        hist, x_edges, y_edges = np.histogram2d(samples[:, 0], samples[:, 1], bins=(bins_x, bins_y), density=True)
        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]  # Keep only non-zero probabilities

        entropy_value = entropy(hist_flat, base=2)
        entropy_estimate = entropy_value #- np.log2(dx * dy) 

        return entropy_estimate

    except (ValueError, ZeroDivisionError) as e:
        print(f"Error in entropy estimation: {e}")
        return None  # Or handle the error as needed (e.g., return a default value)

def _determine_bins(samples, method):
    """
    Determines the optimal number of bins for histogram-based entropy estimation.

    Args:
        samples: A NumPy array of shape (num_samples, 2) containing the samples.
        method: 'scott' or 'freedman' for bin width selection.

    Returns:
        A tuple (bins_x, bins_y) with the number of bins for each dimension.

    Raises:
        ValueError: If bin width calculation fails or results in invalid values.
    """
    n_samples = len(samples)
    range_x = np.ptp(samples[:, 0])
    range_y = np.ptp(samples[:, 1])
    if method == 'scott':
        std_dev_x = np.std(samples[:, 0])
        std_dev_y = np.std(samples[:, 1])

        if std_dev_x == 0 or std_dev_y == 0:
            raise ValueError("Standard deviation of one or both dimensions is zero.")

        bin_width_x = 3.5 * std_dev_x * n_samples**(-1/3) 
        bin_width_y = 3.5 * std_dev_y * n_samples**(-1/3)

        bins_x = max(int(range_x / bin_width_x), 1)
        bins_y = max(int(range_y / bin_width_y), 1)

    elif method == 'freedman':
        iqr_x = np.percentile(samples[:, 0], 75) - np.percentile(samples[:, 0], 25)
        iqr_y = np.percentile(samples[:, 1], 75) - np.percentile(samples[:, 1], 25)

        if iqr_x == 0 or iqr_y == 0:
            raise ValueError("Interquartile range of one or both dimensions is zero.")

        bin_width_x = 2 * iqr_x / n_samples**(1/3)
        bin_width_y = 2 * iqr_y / n_samples**(1/3)

        bins_x = max(int(range_x / bin_width_x), 1)
        bins_y = max(int(range_y / bin_width_y), 1)

    elif method == 'auto':
        try:
            scott_bins = _determine_bins(samples, 'scott')
        except ValueError as e:
            print(f"Scott's rule failed: {e}")
            bins_x, bins_y = _determine_bins(samples, 'freedman') 
        else:
            try:
                freedman_bins = _determine_bins(samples, 'freedman')
            except ValueError as e:
                print(f"Freedman-Diaconis rule failed: {e}")
                bins_x, bins_y = scott_bins
            else:
                if 5 <= min(scott_bins) <= 50 and 5 <= min(freedman_bins) <= 50:
                    # Use Scott's rule if both methods give reasonable bin counts
                    bins_x, bins_y = scott_bins
                elif 5 <= min(scott_bins) <= 50:
                    # Use Scott's rule if it gives a reasonable bin count
                    bins_x, bins_y = scott_bins
                else:
                    # Use Freedman-Diaconis rule if Scott's rule gives unreasonable bin counts
                    bins_x, bins_y = freedman_bins

    return bins_x, bins_y
#
def estimate_entropy_2d_histogram_scipy(samples, bins=100):
  """
  Estimates the entropy of a 2D distribution using a histogram and scipy.stats.entropy.

  Args:
    samples: A NumPy array of shape (num_samples, 2) containing the samples.
    bins: Number of bins for each dimension of the histogram.

  Returns:
    An estimate of the differential entropy of the distribution.
  """

  hist, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, density=True)
  hist_flat = hist.flatten() 
  hist_flat = hist_flat[hist_flat > 0]  # Keep only non-zero probabilities

  # Calculate entropy using scipy.stats.entropy
  entropy_value = entropy(hist_flat, base=2)  # Calculate entropy in bits

  # Calculate bin areas
  dx = 1.0 / bins 
  dy = 1.0 / bins 
  bin_area = dx * dy

  # Adjust entropy for bin size
  adjusted_entropy = entropy_value - np.log2(bin_area) 

  return adjusted_entropy
#
def estimate_entropy_kde(samples, bandwidth=None):
  """
  Estimates the entropy of a 2D distribution using Kernel Density Estimation.

  Args:
    samples: A NumPy array of shape (num_samples, 2) containing the samples.
    bandwidth: Bandwidth parameter for the KDE. If None, uses Scott's rule.

  Returns:
    An estimate of the differential entropy of the distribution.
  """
  kd = kde.gaussian_kde(samples.T, bw_method=bandwidth) 
  # Evaluate the PDF on a fine grid
  x, y = np.mgrid[-5:5:100j, -5:5:100j] 
  positions = np.vstack([x.ravel(), y.ravel()])
  pdf = np.reshape(kd(positions).T, x.shape) 

  # Estimate entropy using numerical integration (trapezoidal rule)
  dx = x[0, 1] - x[0, 0]
  dy = y[1, 0] - y[0, 0]
  entropy_estimate = -np.sum(pdf[pdf > 0] * np.log2(pdf[pdf > 0])) * dx * dy

  return entropy_estimate
#
def estimateKDE_entropy(data):
  """Estimates the entropy of a dataset using KDE.

  Args:
    data: A numpy array containing the data.
    bandwidth: The bandwidth of the KDE kernel. If None, a default bandwidth is used.

  Returns:
    The estimated entropy, or None if an error occurs.
  """

  try:
    kde = stats.gaussian_kde(data, bw_method=None)
    x_grid = np.linspace(min(data), max(data), 1000)
    pdf = kde.evaluate(x_grid)

    entropy = -np.sum(pdf * np.log2(pdf + 1e-10))
    return entropy

  except LinAlgError:
    print("Error: Singular matrix encountered. Consider adjusting bandwidth or using a different KDE implementation.")
    return None

def histogram_entropy_multivariate(data, bins=10):
    """
    Estimate entropy using a histogram-based method for multivariate data.
    
    Args:
        data (array-like): Data sample (n_samples x n_features).
        bins (int or sequence): Number of bins (int) or bins per axis (list of ints).
    
    Returns:
        float: Estimated entropy.
    """
    hist, edges = np.histogramdd(data, bins=bins, density=True)  # Multidimensional histogram
    probabilities = hist / np.sum(hist)  # Normalize to probabilities
    probabilities = probabilities[probabilities > 0]  # Ignore empty bins
    return entropy(probabilities, base=2)  # Base-2 entropy (bits)

def knn_entropy(sample, k=3):
    """
    Estimate entropy using k-Nearest Neighbors (kNN).
    
    Args:
        sample (array-like): Data sample (n_samples x n_features).
        k (int): Number of nearest neighbors.
    
    Returns:
        float: Estimated entropy.
    """
    sample = np.atleast_2d(sample)  # Ensure 2D array
    n, d = sample.shape
    # print(f"n = {n}, d = {d}")
    if d == 0:
        raise ValueError("Data dimensionality (d) cannot be zero.")

    # Volume of unit ball in d-dimensions
    try:
        volume_unit_ball = np.pi ** (d / 2) / gamma(d / 2 + 1)
    except OverflowError:
        raise ValueError("Dimensionality is too high for stable volume calculation.")

    # Build a k-d tree for efficient neighbor search
    tree = cKDTree(sample)
    
    # Find the distance to the k-th nearest neighbor
    distances, _ = tree.query(sample, k=k+1)  # k+1 because query includes the point itself
    kth_distances = distances[:, -1]

    # Prevent division by zero or log(0)
    if np.any(kth_distances <= 0):
        #raise ValueError("Data contains duplicate points or very small distances.")
        return None

    # kNN entropy formula
    entropy = (
        -np.mean(np.log(kth_distances))
        + np.log(volume_unit_ball)
        + np.log(n - 1)
        - np.log(k)
    )
    return entropy
    
def estimateEntropy(data):
    # knn_entropy_estimate = knn_entropy(data, k=5)
    # print(f"kNN-based entropy estimate: {knn_entropy_estimate} bits")
    hist_entropy = histogram_entropy_multivariate(data, bins=50)
    print(f"Multivariate histogram-based entropy estimate: {hist_entropy:.4f} bits")
    # estimated_entropy = estimateKDE_entropy(data)
    # print(estimated_entropy)