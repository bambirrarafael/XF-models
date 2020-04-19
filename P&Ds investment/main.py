import numpy as np
import sobol_seq
import scipy.optimize as opt
import matplotlib.pyplot as plt
from objectiveFunctions import max_scalability
from objectiveFunctions import min_scalability
from objectiveFunctions import max_impact
from objectiveFunctions import min_impact
from objectiveFunctions import max_financial_return
from objectiveFunctions import min_financial_return
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
n_fo = 3
n_scen = 7
cs = np.array([[(0.35, 0.4), (0.5, 0.6), (0.2, 0.3), (0.3, 0.35)],
               [(0.25, 0.3), (0.3, 0.4), (0.15, 0.2), (0.05, 0.1)],
               [(8.6, 9.5), (8.5, 10), (10, 12), (8.5, 9.8)]])
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
    return 1700 - np.sum(x)
cons = [{'type': 'ineq', 'fun': constraint}]
x0 = np.zeros(n_var)    # initial point for optimization
bounds = [(0, 400), (0,  550), (0, 650), (0, 500)]
#
# find 2*n + 1 solutions for all scenarios
mono_obj_sol = np.zeros([n_fo, 2])
harm_sol = np.zeros([n_scen, n_var])
a = [max_scalability, max_impact, max_financial_return]
a_inv = [min_scalability, min_impact, min_financial_return]
goal = [max_scalability, min_impact, max_financial_return]
"""for i in range(n_scen):
    for k in range(n_fo):
        mono_obj_sol[k, 0] = opt.minimize(a[k], x0=x0, args=(estates_of_nature, i), constraints=cons, bounds=bounds,
                                          method=opt_meth).fun
        mono_obj_sol[k, 1] = opt.minimize(a_inv[k], x0=x0, args=(estates_of_nature, i), constraints=cons, bounds=bounds,
                                          method=opt_meth).fun
    x0inp = opt.minimize(goal[0], x0=x0, args=(estates_of_nature, i), constraints=cons, bounds=bounds).x
    harm_sol[i, :] = opt.minimize(maxmin, x0=x0inp, args=(estates_of_nature, i, n_fo, mono_obj_sol), constraints=cons,
                                  bounds=bounds, method=opt_meth).x"""

A = [np.ones(n_var)]
b = 1700
for i in range(n_scen):
    for k in range(n_fo):
        c = estates_of_nature[i, :, k]
        # c = list(c)
        if k == 1:
            mono_obj_sol[k, 0] = opt.linprog(c, A_ub=A, b_ub=b, bounds=bounds).fun
            mono_obj_sol[k, 1] = opt.linprog(-c, A_ub=A, b_ub=b, bounds=bounds).fun
        else:
            mono_obj_sol[k, 0] = opt.linprog(-c, A_ub=A, b_ub=b, bounds=bounds).fun
            mono_obj_sol[k, 1] = opt.linprog(c, A_ub=A, b_ub=b, bounds=bounds).fun
    x0inp = opt.linprog(estates_of_nature[0, :, 0], A_ub=A, b_ub=b, bounds=bounds).x
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
payoff_scalability = -payoffs[0]
payoff_impact = payoffs[1]
payoff_financial_return = -payoffs[2]
print(' ----- payoff scalability ----- ')
print(payoff_scalability)
print(' ----- payoff impact ----- ')
print(payoff_impact)
print(' ----- payoff financial return ----- ')
print(payoff_financial_return)
print('\n')
#
# ============= <X, F> Model ===========
alpha = 0.75
#
# build regret matrix
r_scalability = build_regret_matrix(payoff_scalability)
r_impact = build_regret_matrix(payoff_impact)
r_financial_return = build_regret_matrix(payoff_financial_return)
print(' -----regret scalability ----- ')
print(r_scalability)
print(' ----- regret impact ----- ')
print(r_scalability)
print(' ----- regret financial return ----- ')
print(r_scalability)
print('\n')
#
cc_scalability = build_choice_criteria_matrix(r_scalability, alpha)
cc_impact = build_choice_criteria_matrix(r_impact, alpha)
cc_financial_return = build_choice_criteria_matrix(r_financial_return, alpha)
print(' ----- choice criteria scalability ----- ')
print(cc_scalability)
print(' ----- choice criteria  impact ----- ')
print(cc_impact)
print(' ----- choice criteria  financial return ----- ')
print(cc_financial_return)
print('\n')
#
ncc_scalability = build_normalized_choice_criteria_matrix(cc_scalability)
ncc_impact = build_normalized_choice_criteria_matrix(cc_impact)
ncc_financial_return = build_normalized_choice_criteria_matrix(cc_financial_return)
print(' ----- normalized choice criteria scalability ----- ')
print(ncc_scalability)
print(' ----- normalized choice criteria  impact ----- ')
print(ncc_impact)
print(' ----- normalized choice criteria  financial return ----- ')
print(ncc_financial_return)
print('\n')
#
result = np.zeros([n_alt, 4])
for i in range(n_alt):
    for j in range(4):
        result[i, j] = np.min([ncc_scalability[i, j], ncc_impact[i, j], ncc_financial_return[i, j]])
#
print('---------- RESULT -----------')
print('\n')
print(result)
print('\n')
print('----------- END -------------')

