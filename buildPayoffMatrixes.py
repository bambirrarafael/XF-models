import numpy as np
from objectiveFunctions import npv_cost
from objectiveFunctions import vol_cost
from objectiveFunctions import vol_gfis
from objectiveFunctions import res_diversity
from portfolioParameters import Parameters

pr = Parameters()


def build_payoff_cost(sol_harmonious, n_sol, n_scen):
    poffm_cost = np.zeros([n_sol, n_scen])
    for i in range(n_sol):
        for j in range(n_scen):
            poffm_cost[i, j] = npv_cost(sol_harmonious[i], j)
    return poffm_cost


def build_payoff_vol_cost(sol_harmonious, n_sol, n_scen):
    poffm = np.zeros([n_sol, n_scen])
    for i in range(n_sol):
        for j in range(n_scen):
            poffm[i, j] = vol_cost(sol_harmonious[i], j)
    return poffm


def build_payoff_vol_gfis(sol_harmonious, n_sol, n_scen):
    poffm = np.zeros([n_sol, n_scen])
    for i in range(n_sol):
        for j in range(n_scen):
            poffm[i, j] = vol_gfis(sol_harmonious[i], j)
    return poffm


def build_payoff_div(sol_harmonious, n_sol, n_scen):
    poffm = np.zeros([n_sol, n_scen])
    for i in range(n_sol):
        for j in range(n_scen):
            poffm[i, j] = res_diversity(sol_harmonious[i])
    return poffm


def add_sol_nothing_2_obj(p1, p2, n_scen):
    p1n = np.zeros([n_scen + 1, n_scen])
    p2n = np.zeros([n_scen + 1, n_scen])
    for j in range(n_scen):
        p1n[0, j] = npv_cost(np.zeros([pr.T * pr.n]), j)
        p2n[0, j] = vol_cost(np.zeros([pr.T * pr.n]), j)
    for i in range(n_scen):
        p1n[i + 1, :] = p1[i, :]
        p2n[i + 1, :] = p2[i, :]
    return p1n, p2n


def add_sol_nothing(p1, p2, p3, p4, n_scen):
    p1n = np.zeros([n_scen + 1, n_scen])
    p2n = np.zeros([n_scen + 1, n_scen])
    p3n = np.zeros([n_scen + 1, n_scen])
    p4n = np.zeros([n_scen + 1, n_scen])
    for j in range(n_scen):
        p1n[0, j] = npv_cost(np.zeros([pr.T * pr.n]), j)
        p2n[0, j] = vol_cost(np.zeros([pr.T * pr.n]), j)
        p3n[0, j] = vol_gfis(np.zeros([pr.T * pr.n]), j)
        p4n[0, j] = res_diversity(np.zeros([pr.T * pr.n]))
    for i in range(n_scen):
        p1n[i + 1, :] = p1[i, :]
        p2n[i + 1, :] = p2[i, :]
        p3n[i + 1, :] = p3[i, :]
        p4n[i + 1, :] = p4[i, :]
    return p1n, p2n, p3n, p4n