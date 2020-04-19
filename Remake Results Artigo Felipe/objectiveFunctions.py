import numpy as np


def min_supply(x, est, s):  # goal: min
    return np.sum(est[s, :, 0] * x)


def max_supply(x, est, s):
    return -np.sum(est[s, :, 0] * x)


def min_profit(x, est, s):  # goal: max
    return np.sum(est[s, :, 1] * x)


def max_profit(x, est, s):
    return -np.sum(est[s, :, 1] * x)


def maxmin(x, est, s, n_fo, mono_obj_sol):
    mu = np.zeros(n_fo)
    mu[0] = (mono_obj_sol[0, 0] - min_supply(x, est, s)) / (mono_obj_sol[0, 0] - mono_obj_sol[0, 1])            # min
    mu[1] = (mono_obj_sol[1, 0] - min_profit(x, est, s)) / (mono_obj_sol[1, 0] - mono_obj_sol[1, 1])            # min
    d = np.min(mu)
    return -d