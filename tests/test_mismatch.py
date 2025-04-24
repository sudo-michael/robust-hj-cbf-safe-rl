# %%
from odp.Grid.GridProcessing import Grid
from odp.compute_trajectory import compute_opt_traj
from odp.dynamics.DubinsCar import DubinsCar
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as anim

grid = Grid(
    np.array([-4, -4, -np.pi]),
    np.array([4, 4, np.pi]),
    3,
    np.array([101, 101, 101]),
    [2],
)
car_r = 0.2

car_f1 = DubinsCar(uMode="max", dMode="min", speed=1, wMax=0.5)
car_f2 = DubinsCar(uMode="max", dMode="min", speed=2, wMax=0.5)

EPS = 1 

# %%
target_r = 0.5 + car_r

from odp.Shapes.ShapesFunctions import *
from odp.Plots.plot_options import *
from odp.solver import HJSolver

ivf = CylinderShape(grid, [2], np.zeros(3), target_r)

compMethods = {"TargetSetMode": "minVWithV0"}
# %%
lookback_length = 2
t_step = 0.05
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

#%%
result = HJSolver(
    car_f1,
    grid,
    ivf,
    tau,
    compMethods,
    PlotOptions(
        do_plot=False, plot_type="set", plotDims=[0, 1, 2], slicesCut=[]
    ),
    saveAllTimeSteps=False,
    untilConvergent=False,
    epsilon=0.000005
)

np.save("./brt_v1.npy", result)

result = HJSolver(
    car_f2,
    grid,
    ivf,
    tau,
    compMethods,
    PlotOptions(
        do_plot=False, plot_type="set", plotDims=[0, 1, 2], slicesCut=[]
    ),
    saveAllTimeSteps=False,
    untilConvergent=False,
    epsilon=0.000005
)

np.save("./brt_v2.npy", result)
# %%
brt_v1 = np.load("./brt_v1.npy")
brt_v2 = np.load("./brt_v2.npy")









# %%
def create_render_frame(traj, brt):
    def render_frame(i, colorbar=False):
        # target set 
        plt.contour(grid.grid_points[0],
                    grid.grid_points[1],
                    ivf[:, :, 0].T,
                    levels=0,
                    colors="green",
                    linewidths=1)

                    
        # reachable set at t=0
        index = grid.get_index(traj[0])
        plt.contour(grid.grid_points[0],
                    grid.grid_points[1],
                    brt[:, :, index[2], 0].T,
                    levels=0,
                    colors="black",
                    linewidths=3)

        # trajectory at current time
        plt.plot(traj[i][0], traj[i][1], 'o')
        plt.xlabel('x')
        plt.ylabel('y')
    return render_frame
# %%
traj_car_f1_brt_f1, _, _, _ = compute_opt_traj(car_f1, grid, brt_f1, tau, np.array([-1.5, -1.5, np.pi/4]))
traj_car_f2_brt_f2, _, _, _ = compute_opt_traj(car_f2, grid, brt_f2, tau, np.array([-1.5, -1.5, np.pi/4]))

traj_car_f1_brt_f2, _, _, _ = compute_opt_traj(car_f1, grid, brt_f2, tau, np.array([-1.5, -1.5, np.pi/4]))
traj_car_f2_brt_f1, _, _, _ = compute_opt_traj(car_f2, grid, brt_f1, tau, np.array([-1.5, -1.5, np.pi/4]))
# %%
fig = plt.figure(figsize=(13, 8))
render_frame = create_render_frame(traj_car_f1_brt_f1, brt_f1)
render_frame(0, True)
animation = HTML(anim.FuncAnimation(fig, render_frame, result.shape[-1], interval=40).to_html5_video())
fig.suptitle('Dyn f1, Brt f1')
plt.close(); 
animation
# %%
fig = plt.figure(figsize=(13, 8))
fig.suptitle('Dyn f2, Brt f2')
render_frame = create_render_frame(traj_car_f2_brt_f2, brt_f2)
render_frame(0, True)
animation = HTML(anim.FuncAnimation(fig, render_frame, result.shape[-1], interval=40).to_html5_video())
plt.close(); 
animation
# %%
fig = plt.figure(figsize=(13, 8))
fig.suptitle('Dyn f1, Brt f2')
render_frame = create_render_frame(traj_car_f1_brt_f2, brt_f2)
render_frame(0, True)
animation = HTML(anim.FuncAnimation(fig, render_frame, result.shape[-1], interval=40).to_html5_video())
plt.close(); 
animation

# %%
fig = plt.figure(figsize=(13, 8))
fig.suptitle('Dyn f2, Brt f1')
render_frame = create_render_frame(traj_car_f2_brt_f1, brt_f1)
render_frame(0, True)
animation = HTML(anim.FuncAnimation(fig, render_frame, result.shape[-1], interval=40).to_html5_video())
plt.close(); 
animation

# %%
# what is the difference in hamiltonian?


