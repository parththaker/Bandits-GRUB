"""
Supporting Function Library

Contains function definitions required for a more generic use-case.
"""


import numpy as np


def matrix_norm(x, V):
    """
    Compute quadratic function value <x, Vx>.

    Parameters
    ----------
    x : vector
    V : Matrix

    Returns
    -------
    float : quadratic function value <x, Vx>
    """

    return np.dot( x,np.dot(V, x))


def gaussian_reward(mu_i, mag=1.0):
    """
    Generate gaussian rewards given mean and variance. This is only for 1-dim case.

    Parameters
    ----------
    mu_i : mean value
    mag : variance value

    Returns
    -------
    float : gaussian reward with given mean and variance
    """

    return mu_i + np.random.randn()*mag


def round_function(i):
    """
    Number of arms to be played in a round before Estimation/Elimination routine.

    Parameters
    ----------
    i : round number

    Returns
    -------
    int : number of arms to be played in the current round
    """

    # return 2*i
    # return 2**i
    return 1
