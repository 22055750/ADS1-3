"""
Module to replace the scipy.stats functions skew(), kurtosis() and bootstrap().
Imports numpy.

@author: napi
"""

""" Module to provide three functions:
    skew to compute the skewness of a distribution
    kurtosis to compute the kurtosis of a distribution
    bootstrap to bootstrap errors of statistical values, like std. dev
"""

import numpy as np


def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """
    
    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    
    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / len(dist-2)
    
    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """
    
    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    
    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-3) - 3.0
    
    return value


def bootstrap(dist, function, confidence_level=0.90, nboot=10000):
    """ Carries out a bootstrap of dist to get the uncertainty of statistical
    function applied to it. Dist can be a numpy array or a pandas dataseries.
    confidence_level specifies the quantile (defaulted to 0.90). E.g 0.90
    means the quantile from 0.05 to 0.95 is evaluated. confidence_level=0.682
    gives the range corresponding to 1 sigma, but evaluated using the 
    corresponding quantiles.
    nboot (default 10000) is the number of bootstraps to be evaluated. 
    Returns the lower and upper quantiles. 
    A call of the form
    low, high = bootstrap(dist, np.mean, confidence_level=0.682)
    will return the lower and upper limits of the 1 sigma range"""
    
    fvalues = np.array([]) # creates an empty array to store function values
    dlen = len(dist)
    for i in range(nboot):
        rand = np.random.choice(dist, dlen, replace=True)
        f = function(rand)
        fvalues = np.append(fvalues, f)
        
    # lower and upper quantiles
    qlow = 0.5 - confidence_level/2.0
    qhigh = 0.5 + confidence_level/2.0

    low = np.quantile(fvalues, qlow)
    high = np.quantile(fvalues, qhigh)
    
    return low, high


# checks whether module is imported or run directly.
# code is not executed if imported
if __name__ == "__main__":
    
    dist = np.random.normal(4.0, 3.0, 10000)
    
    print("skewness =", np.round(skew(dist), 6))
    print("kurtosis =", np.round(kurtosis(dist), 6))
    
    print()
    # Call the boostrap routine with statistical functions
    low, high = bootstrap(dist, np.mean, confidence_level=0.682)
    sigma = 0.5 * (high - low)
    print("average = ", np.round(np.mean(dist), 4), "+/-", np.round(sigma, 4))
    
    low, high = bootstrap(dist, np.std, confidence_level=0.682)
    sigma = 0.5 * (high - low)
    print("std. dev = ", np.round(np.std(dist), 4), "+/-", np.round(sigma, 4))
    
    low, high = bootstrap(dist, skew, confidence_level=0.682)
    sigma = 0.5 * (high - low)
    print("skewness = ", np.round(skew(dist), 4), "+/-", np.round(sigma, 4))
    
    low, high = bootstrap(dist, kurtosis, confidence_level=0.682)
    sigma = 0.5 * (high - low)
    print("kurtosis = ", np.round(kurtosis(dist), 4), "+/-", 
          np.round(sigma, 4))



    

