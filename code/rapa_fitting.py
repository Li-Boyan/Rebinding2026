"""
Usage:
    rapa_fitting.py <input_file> [--tol=<tol>] [--bootstrap]

Options:
  -h --help     Show this screen.
"""
import numpy as np
from scipy.integrate import solve_ivp
from rebinding.efflux_model_collection import washout_zero_bound

# from sklearn.metrics import r2_score
import pandas as pd
from dataclasses import dataclass
import yaml
from scipy.optimize import least_squares
import os
import uuid
from docopt import docopt
from tx_fitting import site_occupancy
from scipy.stats import pearsonr


@dataclass
class ModelParams(object):
    c0_1: float = 56
    c0_2: float = 7
    c0_3: float = 23
    c0_4: float = 3
    kon: float = 300
    kd: float = 1.5e-2
    m1: float = 0.7
    m2: float = 3
    mu: float = 5

    def __post_init__(self):
        self.koff = self.kd * self.kon


def load_rapa_washout_data():
    data = pd.read_csv("../data/BYL261/BYL261_results.csv")
    t_eval = data.t.unique()
    y_data_1 = data[
        (data.group == "(-) Everolimus") & (data.Dox == "(+) Dox")
    ].c_cell.values.reshape(-1, len(t_eval), order="F")
    y_data_2 = data[
        (data.group == "(-) Everolimus") & (data.Dox == "(-) Dox")
    ].c_cell.values.reshape(-1, len(t_eval), order="F")
    y_data_3 = data[
        (data.group == "(+) Everolimus") & (data.Dox == "(+) Dox")
    ].c_cell.values.reshape(-1, len(t_eval), order="F")
    y_data_4 = data[
        (data.group == "(+) Everolimus") & (data.Dox == "(-) Dox")
    ].c_cell.values.reshape(-1, len(t_eval), order="F")
    return y_data_1, y_data_2, y_data_3, y_data_4, t_eval


def cost_model(y0, args, t_eval, y_data, time_factor=None, return_r=False):
    sol = solve_ivp(
        washout_zero_bound,
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

    # Calculate R score
    if return_r:
        r = pearsonr(
            np.tile(np.log(y_model), 2),
            np.log(y_data[:, : len(y_model)]).flatten(),
        )
        r = r[0]
    return (cost, r) if return_r else cost


def cost_fn(params, y_data, c0, t_eval, time_factor, factor, xc=False, return_r=False):
    if type(params) != list:
        params = params.tolist()

    kon, koff, m1, m2, mu = params
    kd = koff / kon
    costs = [0] * 4
    input_params = [
        [kon, koff, m2, mu],
        [kon, koff, m1, mu],
        [0, koff, m2, mu],
        [0, koff, m1, mu],
    ]
    c0s = input_params[0][:4]
    cost_list = []
    # time_factor = [
    #     [1, 1, 1, 1, 1, 2],
    #     [1, 1, 1, 1, 0.5, 2],
    #     [1, 1, 1, 2, 0, 0],
    #     [1, 1, 1, 2, 0, 0],
    # ]
    for i in range(4):
        c0_curr = c0[i]
        m_curr = input_params[i][2]
        s0 = site_occupancy(c0_curr, m_curr, kd)
        y0 = [c0_curr, s0]
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


def optimize(input_file, tol, bootstrap=False, default_init=True):
    (y_data_1, y_data_2, y_data_3, y_data_4, t_eval) = load_rapa_washout_data()
    if not bootstrap:
        y_data = [y_data_1, y_data_2, y_data_3, y_data_4]
    else:
        y_data = []
        for data in [y_data_1, y_data_2, y_data_3, y_data_4]:
            for j in range(data.shape[1]):
                idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
                data[:, j] = data[idx, j]
            y_data.append(data)
    t_eval = [t_eval] * 4
    if input_file == "default":
        model = ModelParams()
        factor = [1, 1, 2, 2]
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
    c0 = [model.c0_1, model.c0_2, model.c0_1, model.c0_2]
    params = [
        model.kon,
        model.koff,
        model.m1,
        model.m2,
        model.mu,
    ]
    bounds = (1e-5, np.inf)
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
    # res = [params, param_se]
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
