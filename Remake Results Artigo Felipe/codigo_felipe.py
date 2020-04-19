import numpy as np
import sobol_seq
import math
from scipy.optimize import minimize
from scipy.optimize import linprog
from heuristica_relogio import heuristica_relogio
import warnings
from buildPayoffMatrixes import build_payoff_matrix
from XFModel import build_regret_matrix
from XFModel import build_choice_criteria_matrix
from XFModel import build_normalized_choice_criteria_matrix
from objectiveFunctions import min_supply
from objectiveFunctions import min_profit


class f(object):
    """
    Função Objetivo genérica

    Classe que representa uma FO genérica, para implementação do modelo <X,F>
    """

    def __init__(
            self,
            solver,
            goal,
            f_max=None,
            x_max=None,
            f_min=None,
            x_min=None
    ):

        self._solver = solver  # Equação da função
        if goal == "max" or goal == "min":  # Objetivo
            self.goal = goal
        else:
            raise Exception("goal deve ser 'max' ou 'min'.")
        self.f_max = f_max  # Valoes a serem calculados para mu
        self.x_max = x_max
        self.f_min = f_min
        self.x_min = x_min

    def solve(self, x, inv=False):  # Retorna valor de f(X)
        x = np.array(x)
        return self._solver(x) * -1 if inv else self._solver(x)

    def mu(self, x):  # Cálculo de mu, conforme definição
        x = np.array(x)
        if self.goal == "max":
            return (
                    (self.solve(x) - self.f_min)
                    / (self.f_max - self.f_min)
            )
        elif self.goal == "min":
            return (
                    (self.f_max - self.solve(x))
                    / (self.f_max - self.f_min)
            )

class f_linear(f):
    def __init__(self, coefs, goal):
        self.coefs = np.array(coefs)
        f.__init__(self, lambda x: np.dot(self.coefs, x), goal)

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

A_eq = np.array([np.ones(n_var)])
b_eq = np.array([6927])
bounds = [(0, 2512), (0,  1398), (0, 1976), (0, 2910)]
go = [min_supply, min_profit]
harm_sol = np.zeros([n_scen, n_var])
for i in range(n_scen):
    f0 = f_linear(estates_of_nature[i, :, 0], "min")
    f1 = f_linear(estates_of_nature[i, :, 1], "min")

    F = [f0, f1]

    for f_ in F:
        res = linprog(f_.coefs * -1,
                      A_eq=A_eq,
                      b_eq=b_eq,
                      bounds=bounds,
                      method='simplex')
        if res.success:
            f_.f_max = -1 * res.fun
            f_.x_max = res.x
        else:
            warnings.warn("Otimização mal sucedida.")
    for f_ in F:
        res = linprog(f_.coefs,
                      A_eq=A_eq,
                      b_eq=b_eq,
                      bounds=bounds,
                      method='simplex')
        if res.success:
            f_.f_min = res.fun
            f_.x_min = res.x
        else:
            warnings.warn("Otimização mal sucedida.")

    def muD_max(F, x):
        x = np.array(x)
        return np.min([f_.mu(x) for f_ in F])


    def muD_min(x):
        x = np.array(x)
        return np.max([f_.mu(x) * -1 for f_ in F])


    X = heuristica_relogio(F, f1.x_min, bounds, 0.01)
    harm_sol[i, :] = X

#
# remove duplicate solutions
unique_sol = np.unique(harm_sol, axis=0)
n_alt = len(unique_sol)
print(' ----- unique solution ----- ')
print(unique_sol)
print('\n')
#
# build payoffs
payoffs = build_payoff_matrix(unique_sol, n_alt=n_alt, n_scen=n_scen, est=estates_of_nature, goal=go)
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

