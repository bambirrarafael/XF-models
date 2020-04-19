import numpy as np


def build_payoff_matrix(uniq_sol, n_alt, n_scen, est, goal):
    payoff = []
    for i in range(len(goal)):
        payoff.append(np.zeros([n_alt, n_scen]))
    for i in range(len(goal)):
        for j in range(n_alt):
            for k in range(n_scen):
                payoff[i][j, k] = goal[i](uniq_sol[j], est, k)
    return payoff


