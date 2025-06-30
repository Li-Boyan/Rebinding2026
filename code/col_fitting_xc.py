"""
Usage:
    col_fitting.py <input_file> [--tol=<tol>] [--bootstrap]

Options:
  -h --help     Show this screen.
"""
import numpy as np
from scipy.integrate import solve_ivp
from rebinding.efflux_model_collection import washout_zero_bound_extra_compartment
from scipy.stats import pearsonr
import pandas as pd
from dataclasses import dataclass
import yaml
from scipy.optimize import least_squares
import os
import uuid
from docopt import docopt
from tx_fitting import site_occupancy


@dataclass
class ModelParams(object):
    c0_1: float = 4
    c0_2: float = 4
    c0_3: float = 4
    c0_4: float = 4
    koff_1: float = 0.1
    kd_1: float = 0.02
    m: float = 4
    mu_1: float = 5
    k1_1: float = 0
    k2_1: float = 1
    koff_2: float = 2
    kd_2: float = 0.2
    mu_2: float = 5
    k1_2: float = 1e-3
    k2_2: float = 1e-2

    def __post_init__(self):
        self.kon_1 = self.koff_1 / self.kd_1
        self.kon_2 = self.koff_2 / self.kd_2


def load_washout_data():
    data_1 = pd.read_csv("../data/BYL240/BYL240_results.csv")
    data_2 = pd.read_csv("../data/BYL241/BYL241_results.csv")
    t_eval_1 = data_1[np.logical_not(data_1.Competitor)].t.unique()
    t_eval_2 = data_1[data_1.Competitor].t.unique()
    t_eval_3 = data_2[np.logical_not(data_2.Competitor)].t.unique()
    t_eval_4 = data_2[data_2.Competitor].t.unique()
    y_data_1 = data_1[np.logical_not(data_1.Competitor)].c_cell.values.reshape(
        -1, len(t_eval_1), order="F"
    )
    y_data_2 = data_1[data_1.Competitor].c_cell.values.reshape(
        -1, len(t_eval_2), order="F"
    )
    y_data_3 = data_2[np.logical_not(data_2.Competitor)].c_cell.values.reshape(
        -1, len(t_eval_3), order="F"
    )
    y_data_4 = data_2[data_2.Competitor].c_cell.values.reshape(
        -1, len(t_eval_4), order="F"
    )
    return (
        y_data_1,
        y_data_2,
        y_data_3,
        y_data_4,
        t_eval_1,
        t_eval_2,
        t_eval_3,
        t_eval_4,
    )


def cost_model(y0, args, t_eval, y_data, time_factor=None, return_r=False):
    # print(y0, args, t_eval)
    sol = solve_ivp(
        washout_zero_bound_extra_compartment,
        t_span=[0, t_eval[-1]],
        y0=y0,
        args=args,
        t_eval=t_eval,
    )
    y_model = sol.y[0]
    y_model_adjust = y_model / y_data.mean(axis=0)
    y_data_adjust = y_data / y_data.mean(axis=0)
    cost = y_data_adjust - y_model_adjust
    if not time_factor is None:
        cost *= np.array(time_factor)
    cost = cost.flatten()

    # Calculate r score
    if return_r:
        r = pearsonr(
            np.tile(np.log(y_model), 2),
            (np.log(y_data[:, : len(y_model)])).flatten(),
        )[0]
    return (cost, r) if return_r else cost


def cost_fn(params, y_data, c0, t_eval, time_factor, factor, return_r=False):
    if type(params) != list:
        params = params.tolist()

    (
        kon_1,
        koff_1,
        m,
        mu_1,
        k1_1,
        k2_1,
        kon_2,
        koff_2,
        mu_2,
        k1_2,
        k2_2,
    ) = params
    input_params = [
        [kon_1, koff_1, m, mu_1, k1_1, k2_1],
        [0, koff_1, m, mu_1, k1_1, k2_1],
        [kon_2, koff_2, m, mu_2, k1_2, k2_2],
        [0, koff_2, m, mu_2, k1_2, k2_2],
    ]
    cost_list = []
    for i in range(4):
        kon = input_params[i][0] if input_params[i][0] != 0 else input_params[i - 1][0]
        kd = input_params[i][1] / kon
        c0_curr = c0[i]
        f = input_params[i][-2] / input_params[i][-1]
        s0 = site_occupancy(c0_curr, m, kd)
        u0 = (c0_curr - s0) * f / (1 + f)
        cost_list.append(
            cost_model(
                y0=(c0_curr, s0, u0),
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


def optimize(input_file, tol, bootstrap=False):
    (
        y_data_1,
        y_data_2,
        y_data_3,
        y_data_4,
        t_eval_1,
        t_eval_2,
        t_eval_3,
        t_eval_4,
    ) = load_washout_data()
    if not bootstrap:
        y_data = [y_data_1, y_data_2, y_data_3, y_data_4]
    else:
        y_data = []
        for data in [y_data_1, y_data_2, y_data_3, y_data_4]:
            for j in range(data.shape[1]):
                idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
                data[:, j] = data[idx, j]
            y_data.append(data)
    t_eval = [t_eval_1, t_eval_2, t_eval_3, t_eval_4]
    if input_file == "default":
        model = ModelParams()
        factor = [1, 1]
    else:
        with open(input_file) as f:
            params_dict = yaml.safe_load(f)
        model = ModelParams(**params_dict["MODEL"])
        factor = params_dict["WEIGHTS"]
        if "TIME_FACTOR" in params_dict.keys():
            time_factor = (
                np.array(params_dict["TIME_FACTOR"]).reshape(len(y_data), -1).tolist()
            )
        else:
            time_factor = [None] * (len(y_data_1) + len(y_data_2))
    # Set initial concentrations
    model.c0_1 = y_data[0][:, 0].mean()
    model.c0_2 = y_data[1][:, 0].mean()
    model.c0_3 = y_data[2][:, 0].mean()
    model.c0_4 = y_data[3][:, 0].mean()
    c0 = [model.c0_1, model.c0_2, model.c0_3, model.c0_4]
    params = [
        model.kon_1,
        model.koff_1,
        model.m,
        model.mu_1,
        model.k1_1,
        model.k2_1,
        model.kon_2,
        model.koff_2,
        model.mu_2,
        model.k1_2,
        model.k2_2,
    ]
    lower_bounds = [1e-5] * len(params)
    lower_bounds[4] = 0
    lower_bounds[-2] = 0
    bounds = (lower_bounds, np.inf)
    params_fit = least_squares(
        cost_fn,
        params,
        args=(y_data, c0, t_eval, time_factor, factor),
        method="dogbox",
        ftol=tol,
        verbose=2,
        bounds=bounds,
        max_nfev=10000,
    )
    params = params_fit.x.tolist()
    file_suffix = "_bootstrap" if bootstrap else ""
    output_file = input_file.replace(
        ".yaml", f"_res{file_suffix}_{str(uuid.uuid4())[:8]}.npy"
    )
    np.save(output_file, params)


def main():
    args = docopt(__doc__)
    optimize(
        args["<input_file>"],
        float(args["--tol"]),
        args["--bootstrap"],
    )


if __name__ == "__main__":
    main()
