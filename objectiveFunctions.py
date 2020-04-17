import numpy as np
from portfolioCharacteristcs import calc_exposure
from portfolioCharacteristcs import calc_cost
from portfolioParameters import Parameters

pr = Parameters()


def npv_cost(x, s):
    x = x.reshape([pr.n, pr.T])
    exp = calc_exposure(x, s)
    cost = calc_cost(x=x, exp=exp, s=s)
    return np.mean(cost)
    """f1 = 0
    for t in range(pr.T):
        f1 += cost[t]
    return f1/pr.T"""


def vol_cost(x, s):
    x = x.reshape([pr.n, pr.T])
    f2 = 0
    for t in range(pr.T):
        exp = calc_exposure(x, s)
        f2 += 8760 * (pr.sigma_c * (abs(x[0, t]) + pr.g_it[0, t] + pr.v_t[t]) + pr.sigma_spot * abs(exp[t]))
    return f2/pr.T


def vol_gfis(x, s):
    x = x.reshape([pr.n, pr.T])
    f3 = 0
    for t in range(pr.T):
        for i in range(pr.n):
            f3 += pr.theta_i[i] * (x[i, t] + pr.GSF[i, t, s] * pr.g_it[i, t])
    return f3/pr.T


def res_diversity(x):   # OK!
    x = x.reshape([pr.n, pr.T])
    f4 = 0
    for t in range(pr.T):
        m = 0
        for i in range(pr.n):
            m += x[i, t] + pr.g_it[i, t]
        m = m/pr.n
        aux = 0
        for i in range(pr.n):
            aux += (x[i, t] + pr.g_it[i, t] - m) ** 2
        f4 += np.sqrt(aux/pr.n)
    return f4/pr.T


def npv_cost_max(x, s):
    x = x.reshape([pr.n, pr.T])
    exp = calc_exposure(x, s)
    cost = calc_cost(x=x, exp=exp, s=s)
    return -np.mean(cost)
    """f1 = 0
    for t in range(pr.T):
        f1 += cost[t]
    return -f1/pr.T"""


def vol_cost_max(x, s):
    x = x.reshape([pr.n, pr.T])
    f2 = 0
    for t in range(pr.T):
        exp = calc_exposure(x, s)
        f2 += 8760 * (pr.sigma_c * (abs(x[0, t]) + pr.v_t[t]) + pr.sigma_spot * abs(exp[t]))
    return -f2/pr.T


def vol_gfis_max(x, s):
    x = x.reshape([pr.n, pr.T])
    f3 = 0
    for t in range(pr.T):
        for i in range(pr.n):
            f3 += pr.theta_i[i] * (x[i, t] + pr.GSF[i, t, s] * pr.g_it[i, t])
    return -f3/pr.T


def res_diversity_max(x):   # OK!
    x = x.reshape([pr.n, pr.T])
    f4 = 0
    for t in range(pr.T):
        m = 0
        for i in range(pr.n):
            m += x[i, t] + pr.g_it[i, t]
        m = m/pr.n
        aux = 0
        for i in range(pr.n):
            aux += (x[i, t] + pr.g_it[i, t] - m) ** 2
        f4 += np.sqrt(aux/pr.n)
    return -f4/pr.T


def max_min(x, s, solutions, lambd, n_fo): # todo fix 'vals' values - change axis
    if n_fo == 2:
        mu = np.zeros(2)
        mu[0] = ((solutions['Maximum Cost'][0] - npv_cost(x, s)) /
                 (solutions['Maximum Cost'][0] - solutions['Minimum Cost'][0]))**lambd[0]
        mu[1] = ((solutions['Maximum Cost Volatility'][1] - vol_cost(x, s)) /
                 (solutions['Maximum Cost Volatility'][1] - solutions['Minimum Cost Volatility'][1]))**lambd[1]
        d = np.min(mu)
        return -d
    else:
        mu = np.zeros(4)
        mu[0] = ((solutions['Maximum Cost'][0] - npv_cost(x, s)) / (
                    solutions['Maximum Cost'][0] - solutions['Minimum Cost'][0])) ** lambd[0]
        mu[1] = ((solutions['Maximum Cost Volatility'][1] - vol_cost(x, s)) / (
                    solutions['Maximum Cost Volatility'][1] - solutions['Minimum Cost Volatility'][1])) ** lambd[1]
        mu[2] = ((solutions['Maximum Resources Volatility'][2] - vol_gfis(x, s)) /
                 (solutions['Maximum Resources Volatility'][2] -
                  solutions['Minimum Resources Volatility'][2])) ** lambd[2]
        mu[3] = ((solutions['Maximum Resources Diversity'][3] - res_diversity(x)) /
                 (solutions['Maximum Resources Diversity'][3] -
                  solutions['Minimum Resources Diversity'][3])) ** lambd[3]
        d = np.min(mu)
        return -d
