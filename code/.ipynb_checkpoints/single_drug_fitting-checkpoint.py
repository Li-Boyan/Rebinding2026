"""
Usage:
    single_drug_fitting.py <drug> <input_file> [--tol=<tol>] [--bootstrap] [--default-init]

Options:
  -h --help     Show this screen.
"""
import numpy as np
from scipy.integrate import solve_ivp
from rebinding.efflux_model_collection import (
    washout_zero_bound,
    washout_zero_bound_extra_compartment,
)
from scipy.stats import pearsonr
import pandas as pd
from dataclasses import dataclass
import yaml
from scipy.optimize import minimize, least_squares
import os
import uuid
from docopt import docopt
from tx_fitting import site_occupancy


@dataclass
class ModelParams(object):
    c0_1: float = 20
    c0_2: float = 20
    koff: float = 0.1
    kd: float = 1e-2
    m: float = 13
    mu: float = 1
    k1: float = 1
    k2: float = 1
    xc: bool = False

    def __post_init__(self):
        self.kon = self.koff / self.kd


# Load washout data
drug_res_files = {
    "VB": "../data/BYL265/BYL265_results.csv",
    "DTX": "../data/BYL184/BYL184_rerun_results.csv",
    "Ixa": "../data/BYL286/BYL286_results.csv",
    "Trm": "../data/BYL280/BYL280_results.csv",
    "Slm": "../data/BYL279/BYL279_results.csv",
    "Ali": "../data/BYL180/BYL180_results.csv",
    "Isp": "../data/BYL199/BYL199_results.csv",
    "Erlo": "../data/BYL262/BYL262_results.csv",
    "Top": "../data/BYL234/BYL234_results.csv",
    "Col": "../data/BYL240/BYL240_results.csv",
    "CA4": "../data/BYL241/BYL241_results.csv",
}


def load_washout_data(drug):
    data = pd.read_csv(drug_res_files[drug])
    t_eval = data.t.unique()
    y_data_1 = data[np.logical_not(data.Competitor)].c_cell.values.reshape(
        -1, len(t_eval), order="F"
    )
    y_data_2 = data[data.Competitor].c_cell.values.reshape(-1, len(t_eval), order="F")
    return y_data_1, y_data_2, t_eval


def cost_model(y0, args, t_eval, y_data, time_factor=None, return_r=False):
    sol = solve_ivp(
        washout_zero_bound if len(args) == 4 else washout_zero_bound_extra_compartment,
        t_span=[0, t_eval[-1]],
        y0=y0,
        args=args,
        t_eval=t_eval,
        max_step=0.1,
    )
    y_model = sol.y[0]
    y_model_adjust = y_model / y_data.mean(axis=0)
    y_data_adjust = y_data / y_data.mean(axis=0)
    # cost = np.mean((y_data_adjust - y_model_adjust) ** 2, axis=0)
    cost = y_data_adjust - y_model_adjust
    if not time_factor is None:
        cost *= np.array(time_factor)
    cost = cost.flatten()

    # Calculate r score
    if return_r:
        r = pearsonr(
            np.tile(y_model, 2),
            y_data[:, : len(y_model)].flatten(),
        )[0]
    return (cost, r) if return_r else cost


def cost_fn(
    params,
    y_data,
    c0,
    t_eval,
    time_factor,
    factor,
    xc=False,
    return_r=False,
):
    if type(params) != list:
        params = params.tolist()
    # print(params)
    kon, koff, m = params[:3]
    kd = koff / kon
    input_params = [[kon] + params[1:], [0] + params[1:]]
    cost_list = []
    for i in range(2):
        c0_curr = c0[i]
        s0 = site_occupancy(c0_curr, m, kd)
        y0 = [c0_curr, s0]
        if xc:
            k1, k2 = params[4:6]
            f = k1 / k2
            u0 = (c0_curr - s0) * f / (1 + f)
            y0 = [c0_curr, s0, u0]
        cost_list.append(
            cost_model(
                y0=y0,
                args=input_params[i],
                t_eval=t_eval[i],
                y_data=y_data[i],
                return_r=return_r,
                time_factor=time_factor[i],
            )
        )
    r = []
    if not return_r:
        costs = cost_list
    else:
        costs = [cost[0] for cost in cost_list]
        r = [cost[1] for cost in cost_list]

    # Apply weighting factors and calculate total cost
    weighted_costs = np.concatenate([costs[i] * factor[i] for i in range(len(factor))])

    return (weighted_costs, r) if return_r else weighted_costs


def optimize(
    drug,
    input_file,
    tol,
    bootstrap=False,
    default_init=True,
):
    y_data_1, y_data_2, t_eval = load_washout_data(drug)
    if not bootstrap:
        y_data = [y_data_1, y_data_2]
    else:
        y_data = []
        for data in [y_data_1, y_data_2]:
            for j in range(data.shape[1]):
                idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
                data[:, j] = data[idx, j]
            y_data.append(data)
    t_eval = [t_eval] * 2
    if input_file == "default":
        model = ModelParams()
        factor = [1, 1]
    else:
        with open(input_file) as f:
            params_dict = yaml.safe_load(f)
            model_dict = {k: v for k, v in params_dict["MODEL"].items() if k != "kon"}
        model = ModelParams(**model_dict)
        factor = params_dict["WEIGHTS"]
        if "TIME_FACTOR" in params_dict.keys():
            time_factor = (
                np.array(params_dict["TIME_FACTOR"]).reshape(len(y_data), -1).tolist()
            )
        else:
            time_factor = [None] * (len(y_data_1) + len(y_data_2))
    # Set initial concentrations
    if default_init:
        model.c0_1 = y_data[0][:, 0].mean()
        model.c0_2 = y_data[1][:, 0].mean()
    c0 = [model.c0_1, model.c0_2]
    if drug == "VB":
        c0[0] = model.c0_2
    params = [
        model.kon,
        model.koff,
        model.m,
        model.mu,
    ]
    if model.xc:
        print("Using extra compartment model")
        params += [model.k1, model.k2]

    bounds = (1e-5, np.inf)
    params_fit = least_squares(
        cost_fn,
        params,
        args=(y_data, c0, t_eval, time_factor, factor, model.xc),
        method="dogbox",
        ftol=tol,
        verbose=2,
        bounds=bounds,
        max_nfev=10000,
    )
    # Calculate covariance matrix
    # J = params_fit.jac
    # n = y_data_1.shape[1] + y_data_2.shape[1]
    # p = len(params_fit.x)
    # dof = max(0, n - p)
    # residual_variance = params_fit.cost / dof
    # cov_matrix = np.linalg.inv(J.T @ J) * residual_variance
    # param_se = np.sqrt(np.diag(cov_matrix))

    params = params_fit.x.tolist()
    # res = [params, param_se]
    file_suffix = "_bootstrap" if bootstrap else ""
    output_file = input_file.replace(
        ".yaml", f"_res{file_suffix}_{str(uuid.uuid4())[:8]}.npy"
    )
    np.save(output_file, params)


def main():
    args = docopt(__doc__)
    optimize(
        args["<drug>"],
        args["<input_file>"],
        float(args["--tol"]),
        args["--bootstrap"],
        args["--default-init"],
    )


if __name__ == "__main__":
    main()
