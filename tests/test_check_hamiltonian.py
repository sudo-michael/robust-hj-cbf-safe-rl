# %%
from redexp.utils import spa_deriv
from redexp.brts.dubins_3d import grid, car_brt
import numpy as np
X = np.linspace(-4, 4, 101)
Y = np.linspace(-4, 4, 101)
THETA = np.linspace(-np.pi, np.pi, 101)
BRT = np.load("./redexp/brts/static_obstacle_dubins_3d_brt.npy")
R = 0.3 + 0.5
def l(state):
    return state[0]**2 + state[1]**2 - R**2


best_ham = np.inf
positive_ham = []
non_positive_ham = []
for x in X:
    for y in Y:
        for theta in THETA:
            state = np.array([x, y, theta])
            value = grid.get_value(BRT, state)
            index = grid.get_index(state)
            nabla_V = spa_deriv(index, BRT, grid)
            ham = car_brt.ham(state, car_brt.opt_ctrl_non_hcl(0, state, nabla_V), nabla_V)

            if value > 0.01:
                if ham > 0:
                    positive_ham.append(ham)
                else:
                    non_positive_ham.append(ham)
        # best_ham = min(best_ham, ham)

# %%
print('done')
