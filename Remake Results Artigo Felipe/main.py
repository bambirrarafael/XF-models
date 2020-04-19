import numpy as np
import sobol_seq
import scipy.optimize as opt
import matplotlib.pyplot as plt
from objectiveFunctions import min_supply
from objectiveFunctions import max_supply
from objectiveFunctions import min_profit
from objectiveFunctions import max_profit
from objectiveFunctions import maxmin
from buildPayoffMatrixes import build_payoff_matrix
from XFModel import build_regret_matrix
from XFModel import build_choice_criteria_matrix
from XFModel import build_normalized_choice_criteria_matrix

#
# number print format
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
#
# initial parameters
n_var = 4
n_fo = 2
n_scen = 7
cs = np.array([[(20.5, 31.5),   (42.5, 49.5),   (25.5, 33.5),   (10.5, 14.5)],
               [(2210, 2850),   (1850, 2350),   (1700, 1950),   (2340, 2950)]])
opt_meth = 'SLSQP'
#
# create states of nature
points = sobol_seq.i4_sobol_generate(n_var*n_fo, n_scen)
estates_of_nature = np.zeros([n_scen, n_var, n_fo])
for i in range(n_scen):
    col = 0
    for k in range(n_fo):
        for j in range(n_var):
            aux = (cs[k, j, 1] - cs[k, j, 0])*(1 - points[i, col])
            estates_of_nature[i, j, k] = cs[k, j, 1] - aux
            col += 1
#
# Define problem goal, bounds and constraints
def constraint(x):
    return np.sum(x) - 6927
cons = [{'type': 'eq', 'fun': constraint}]
x0 = np.zeros(n_var)    # initial point for optimization
bounds = [(0, 2512), (0,  1398), (0, 1976), (0, 2910)]
#
# find 2*n + 1 solutions for all scenarios
mono_obj_sol = np.zeros([n_fo, 2])
harm_sol = np.zeros([n_scen, n_var])
a = [min_supply, min_profit]
a_inv = [max_supply, max_profit]
goal = [min_supply, min_profit]
'''for i in range(n_scen):
    for k in range(n_fo):
        mono_obj_sol[k, 0] = opt.minimize(a[k], x0=x0, args=(estates_of_nature, i), constraints=cons, bounds=bounds,
                                          method=opt_meth).fun
        mono_obj_sol[k, 1] = opt.minimize(a_inv[k], x0=x0, args=(estates_of_nature, i), constraints=cons, bounds=bounds,
                                          method=opt_meth).fun
    x0inp = opt.minimize(goal[0], x0=x0, args=(estates_of_nature, i), constraints=cons, bounds=bounds).x
    harm_sol[i, :] = opt.minimize(maxmin, x0=x0inp, args=(estates_of_nature, i, n_fo, mono_obj_sol), constraints=cons,
                                  bounds=bounds, method=opt_meth).x'''

A = [np.ones(n_var)]
b = 6927
for i in range(n_scen):
    for k in range(n_fo):
        c = estates_of_nature[i, :, k]
        mono_obj_sol[k, 0] = opt.linprog(c, A_eq=A, b_eq=b, bounds=bounds).fun
        mono_obj_sol[k, 1] = opt.linprog(-c, A_eq=A, b_eq=b, bounds=bounds).fun

    x0inp = opt.linprog(estates_of_nature[0, :, 0], A_eq=A, b_eq=b, bounds=bounds).x
    harm_sol[i, :] = opt.minimize(maxmin, x0=x0inp, args=(estates_of_nature, i, n_fo, mono_obj_sol), constraints=cons,
                                  bounds=bounds, method=opt_meth).x

#
# remove duplicate solutions
unique_sol = np.unique(harm_sol, axis=0)
n_alt = len(unique_sol)
print(' ----- unique solution ----- ')
print(unique_sol)
print('\n')
#
# build payoffs
payoffs = build_payoff_matrix(unique_sol, n_alt=n_alt, n_scen=n_scen, est=estates_of_nature, goal=goal)
payoff_min_supply = payoffs[0]
payoff_min_profit = payoffs[1]
print(' ----- payoff scalability ----- ')
print(payoff_min_supply)
print(' ----- payoff impact ----- ')
print(payoff_min_profit)
print('\n')
#
# ============= <X, F> Model ===========
alpha = 0.75
#
# build regret matrix
r_min_supply = build_regret_matrix(payoff_min_supply)
r_min_profit = build_regret_matrix(payoff_min_profit)
print(' -----regret supply ----- ')
print(r_min_supply)
print(' ----- regret profit ----- ')
print(r_min_profit)
print('\n')
#
cc_min_supply = build_choice_criteria_matrix(r_min_supply, alpha)
cc_min_profit = build_choice_criteria_matrix(r_min_profit, alpha)
print(' ----- choice criteria supply ----- ')
print(cc_min_supply)
print(' ----- choice criteria profit ----- ')
print(cc_min_profit)
print('\n')
#
ncc_min_supply = build_normalized_choice_criteria_matrix(cc_min_supply)
ncc_min_profit = build_normalized_choice_criteria_matrix(cc_min_profit)
print(' ----- normalized choice criteria supply ----- ')
print(ncc_min_supply)
print(' ----- normalized choice criteria profit ----- ')
print(ncc_min_profit)
print('\n')
#
result = np.zeros([n_alt, 4])
for i in range(n_alt):
    for j in range(4):
        result[i, j] = np.min([ncc_min_supply[i, j], ncc_min_profit[i, j]])
#
print('---------- RESULT -----------')
print('\n')
print(result)
print('\n')
print('----------- END -------------')
