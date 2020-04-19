import numpy as np


def max_scalability(x, est, s):  # goal: max
    return -np.sum(est[s, :, 0] * x)


def min_scalability(x, est, s):
    return np.sum(est[s, :, 0] * x)


def min_impact(x, est, s):  # goal: min
    return np.sum(est[s, :, 1] * x)


def max_impact(x, est, s):
    return -np.sum(est[s, :, 1] * x)


def max_financial_return(x, est, s):  # goal: max
    return -np.sum(est[s, :, 2] * x)


def min_financial_return(x, est, s):
    return np.sum(est[s, :, 2] * x)


def maxmin(x, est, s, n_fo, mono_obj_sol):
    mu = np.zeros(n_fo)
    mu[0] = (max_scalability(x, est, s) - mono_obj_sol[0, 0]) / (mono_obj_sol[0, 0] - mono_obj_sol[0, 1])       # max
    mu[1] = (mono_obj_sol[1, 0] - min_impact(x, est, s)) / (mono_obj_sol[1, 0] - mono_obj_sol[1, 1])            # min
    mu[0] = (max_financial_return(x, est, s) - mono_obj_sol[2, 0]) / (mono_obj_sol[2, 0] - mono_obj_sol[2, 1])  # max
    d = np.min(mu)
    return -d