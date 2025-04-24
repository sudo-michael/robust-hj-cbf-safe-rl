# import sys
# sys.path.append("../tmp/deepreach-verification")
import time
import pickle
from redexp.deepreach.dynamics import dynamics
from redexp.deepreach.utils import modules
import os
import torch

import pathlib
# print(pathlib.Path().resolve())


# model_dir = "../tmp/deepreach-verification/runs/atu_dubins"
model_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "brts/deepreach/dubins3d_2/")
opt = os.path.join(model_dir, "orig_opt.pickle")
with open(opt, "rb") as f:
    orig_opt = pickle.load(f)
    for k, v in orig_opt.__dict__.items():
        print(k, v)


dynamics = dynamics.Dubins3D(
    orig_opt.goalR,
    orig_opt.velocity,
    orig_opt.omega_max,
    orig_opt.angle_alpha_factor,
    orig_opt.set_mode,
    orig_opt.diff_model,
    orig_opt.freeze_model,
)

model = modules.SingleBVPNet(
    in_features=dynamics.input_dim,
    out_features=1,
    type=orig_opt.model,
    mode=orig_opt.model_mode,
    final_layer_factor=1.0,
    hidden_features=orig_opt.num_nl,
    num_hidden_layers=orig_opt.num_hl,
)

# benchmark cannot re-init CUDA in forked subprocess
# for inference, cpu prob not needed
# model.cuda()

model_path = os.path.join(model_dir, 'model_final.pth')
model.load_state_dict(torch.load(model_path))

def value(state, device='cpu'):
    # state: np.array 
    coords = torch.zeros(1, dynamics.state_dim + 1, device=device)
    coords[:, 0] = 1
    coords[:, 1:] = torch.from_numpy(state).to(device)

    # coords := world space coordinates
    with torch.no_grad():
        # model_results = model({'coords': dynamics.coord_to_input(coords.cuda())})
        model_results = model({'coords': dynamics.coord_to_input(coords)})
        values = dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
        return values.item()

def opt_action(state, device='cpu'):
    # state: np.array 
    coords = torch.zeros(1, dynamics.state_dim + 1, device=device)
    coords[:, 0] = 1
    coords[:, 1:] = torch.from_numpy(state).to(device)
    # coords := world space coordinates
    # model_results = model({'coords': dynamics.coord_to_input(coords.cuda())})
    model_results = model({'coords': dynamics.coord_to_input(coords)})
    dv = dynamics.io_to_dv(
        model_results["model_in"],
        model_results["model_out"].squeeze(dim=-1),
    ).detach()
    # opt_action = dynamics.optimal_control(coords[:, 1:].cuda(), dv[..., 1:].cuda())
    opt_action = dynamics.optimal_control(coords[:, 1:], dv[..., 1:])
    return opt_action.flatten().cpu().numpy() # (1, )

def true_hamiltonian(state, device='cpu'):
    coords = torch.zeros(1, dynamics.state_dim + 1, device=device)
    coords[:, 0] = 1
    coords[:, 1:] = torch.from_numpy(state).to(device)
    # coords := world space coordinates
    # model_results = model({'coords': dynamics.coord_to_input(coords.cuda())})
    model_results = model({'coords': dynamics.coord_to_input(coords)})
    dv = dynamics.io_to_dv(
        model_results["model_in"],
        model_results["model_out"].squeeze(dim=-1),
    ).detach()
    # opt_action = dynamics.optimal_control(coords[:, 1:].cuda(), dv[..., 1:].cuda())
    breakpoint() 
    ham = dynamics.hamiltonian(coords[:, 1:], dv)
    return ham

if __name__ in "__main__":
    plot_config = dynamics.plot_config()

    # state_test_range = dynamics.state_test_range()
    # x_min, x_max = state_test_range[plot_config['x_axis_idx']]
    # y_min, y_max = state_test_range[plot_config['y_axis_idx']]
    # z_min, z_max = state_test_range[plot_config['z_axis_idx']]

    # times = torch.linspace(0, 1, 3)
    # xs = torch.linspace(x_min, x_max, 200)
    # ys = torch.linspace(y_min, y_max, 200)
    # zs = torch.linspace(z_min, z_max, 5)
    # xys = torch.cartesian_prod(xs, ys)

    # import matplotlib.pyplot as plt

    # fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
    # for i in range(len(times)):
    #     for j in range(len(zs)):
    #         coords = torch.zeros(200*200, dynamics.state_dim + 1)
    #         coords[:, 0] = times[i]
    #         coords[:, 1:] = torch.tensor(plot_config['state_slices'])
    #         coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
    #         coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
    #         coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

    #         with torch.no_grad():
    #             model_results = model({'coords': dynamics.coord_to_input(coords.cuda())})
    #             values = dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
            
    #         ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
    #         ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
    #         s = ax.imshow(1*(values.detach().cpu().numpy().reshape(200, 200).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
    #         fig.colorbar(s) 
    # fig.savefig('new_deepreach.png')

    # test if model is faster on gpu or cpu
    # cpu: 12.9
    # gpu: 22.9
    # import numpy as np
    # print(next(model.parameters()).is_cuda)
    # tik = time.time()
    # for _ in range(10_000):
    #     state = np.random.rand(3)
    #     value(state)
    #     opt_action(state)
    # print("CPU:", time.time() - tik)


    # model.cuda()
    # print(next(model.parameters()).is_cuda)
    # tik = time.time()
    # for _ in range(10_000):
    #     state = np.random.rand(3)
    #     value(state, device='cuda')
    #     opt_action(state, device='cuda')
    # print("GPU:", time.time() - tik)