from odp.Grid.GridProcessing import Grid
from redexp.dynamics.dubins_car import DubinsCar
import numpy as np

grid = Grid(
    np.array([-4, -4, -np.pi]),
    np.array([4, 4, np.pi]),
    3,
    np.array([101, 101, 101]),
    [2],
)
car_r = 0.2

car_brt = DubinsCar(r=car_r, uMode="max", dMode="min", speed=1, wMax=1)

obstacle_r = 0.5
goal_r = 0.3
cylinder_r = car_r + obstacle_r

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(grid, [2], np.zeros(3), cylinder_r)

    lookback_length = 20.0
    t_step = 0.05
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    compMethods = {"TargetSetMode": "minVWithV0"}

    # po2 = PlotOptions(do_plot=True, plot_type="set", plotDims=[0,1,2], slicesCut=[], min_isosurface = 0, max_isosurface = 0, colorscale='Bluered')
    result = HJSolver(
        car_brt,
        grid,
        ivf,
        tau,
        compMethods,
        PlotOptions(
            do_plot=True, plot_type="set", plotDims=[0, 1, 2], slicesCut=[]
        ),
        saveAllTimeSteps=False,
        untilConvergent=True,
        epsilon=0.000005
    )

    np.save("./test_brt.npy", result)
