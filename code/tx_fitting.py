"""
Usage:
    tx_fitting.py <input_file> <output_file> [--tol=<tol>] [--bootstrap] [--default-init] [--fit-init]

Options:
  -h --help     Show this screen.
"""
import numpy as np
from scipy.integrate import solve_ivp
from rebinding.efflux_model_collection import *
from scipy.stats import pearsonr
import pandas as pd
from dataclasses import dataclass
import yaml
from scipy.optimize import minimize
import os
import uuid
from docopt import docopt


@dataclass
class ModelParams(object):
    kon: float = 129.6
    koff: float = 13.1
    m: float = 13
    mu: float = 1.5
    k: float = 50
    c0_1: float = 25
    c0_2: float = 25
    c0_3: float = 25
    c0_4: float = 25
    Vc1: float = 5.25e-3
    V1: float = 1
    Vc2: float = 6.56e-3
    V2: float = 1

    def __post_init__(self):
        self.kd = self.koff / self.kon


def load_washout_once_data():
    # Lysate data
    lysate_data = pd.concat(
        [
            pd.read_csv("../data/BYL118/BYL118_lysate_rb_results.csv"),
            pd.read_csv("../data/BYL118/BYL118_lysate_results.csv"),
        ]
    )
    lysate_data["Competitor"] = np.repeat([False, True], len(lysate_data) // 2)
    lysate_data = lysate_data.sort_values(by="t")
    y_data_1 = lysate_data[np.logical_not(lysate_data.Competitor)][
        "3H-PTX"
    ].values.reshape((3, -1), order="F")
    t_eval = [
        lysate_data[np.logical_not(lysate_data.Competitor)].t.unique(),
        lysate_data[lysate_data.Competitor].t.unique(),
    ]

    y_data_2 = lysate_data[lysate_data.Competitor]["3H-PTX"].values.reshape(
        (3, -1), order="F"
    )

    # Media data
    medium_data = pd.concat(
        [
            pd.read_csv("../data/BYL118/BYL118_medium_rb_results.csv"),
            pd.read_csv("../data/BYL118/BYL118_medium_results.csv"),
        ]
    )
    medium_data["Competitor"] = np.repeat([False, True], len(medium_data) // 2)

    medium_data = medium_data.sort_values(by="t")
    y_data_1 = np.concatenate(
        [
            y_data_1,
            medium_data[np.logical_not(medium_data.Competitor)][
                "3H-PTX"
            ].values.reshape((3, -1), order="F"),
        ],
        axis=1,
    )
    y_data_2 = np.concatenate(
        [
            y_data_2,
            medium_data[medium_data.Competitor]["3H-PTX"].values.reshape(
                (3, -1), order="F"
            ),
        ],
        axis=1,
    )
    return y_data_1, y_data_2, t_eval


def load_multi_wash_data():
    multiwash_data = pd.read_csv("../data/BYL164/BYL164_clean.csv")
    y_data_3 = multiwash_data[np.logical_not(multiwash_data.condition)]["3H-PTX"].values
    y_data_3 = np.array(y_data_3.tolist() + [y_data_3[-2:].mean()])
    y_data_3 = y_data_3.reshape((3, -1), order="F")
    y_data_4 = multiwash_data[multiwash_data.condition]["3H-PTX"].values
    y_data_4 = y_data_4.reshape((3, -1), order="F")

    t_eval = [
        multiwash_data[np.logical_not(multiwash_data.condition)].t.unique(),
        multiwash_data[multiwash_data.condition].t.unique(),
    ]

    return y_data_3, y_data_4, t_eval


def site_occupancy(c0, m, kd):
    """Calculate the initial site occupancy based on given parameters and concentration."""
    a = 1
    b = kd + c0 + m
    c = m * c0
    s_init = (b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
    return s_init


def cost_model(params, model, y0, args, t_eval, y_data, time_factor=None, media=False, return_r=False):
    c0_1, c0_2, c0_3, c0_4, kon, koff, m, mu, k = params
    sol = solve_ivp(
        model,
        t_span=[0, t_eval[-1]],
        y0=y0,
        args=args,
        t_eval=t_eval,
        # max_step=0.1,
    )
    y_model_lysate = sol.y[0]
    y_model_media = sol.y[1] * 1e3
    # media_factor = (
    #     y_data[:, : y_data.shape[1] // 2].max()
    #     / y_data[:, y_data.shape[1] // 2 :].max()
    # )
    # y_model_media *= media_factor
    y_model = (
        np.concatenate([y_model_lysate, y_model_media]) if media else y_model_lysate
    )
    # y_data_adjust = np.log(y_data + 1e-3)
    # y_model_adjust = np.log(y_model + 1e-3)
    y_model_adjust = y_model / y_data.mean(axis=0)
    y_data_adjust = y_data / y_data.mean(axis=0)
    # y_data_adjust = y_data.copy()
    # if media:
    #     y_data_adjust *= np.repeat([1, media_factor], y_data.shape[1] // 2)
    cost = np.mean((y_data_adjust - y_model_adjust) ** 2, axis=0)
    if not time_factor is None:
        cost *= time_factor
    cost = cost.sum()

    # Calculate r score
    if return_r:
        # print(y_model_lysate, y_data[:, : len(y_model_lysate)])
        y_data_lysate = y_data[:, : len(y_model_lysate)]
        lysate_r = pearsonr(
            np.tile(np.log(y_model_lysate + 1e-3), 3),
            (np.log(y_data_lysate + 1e-3)).flatten(),
        )[0]
        r = [lysate_r]
        if media:
            media_r = pearsonr(
                np.tile(y_model_media, 3),
                y_data[:, y_data.shape[1] // 2 :].flatten(),
            )[0]
            r.append(media_r)
    return (cost, r) if return_r else cost


def cost_fn(params, fixed_params, ctrl_params, y_data, t_eval, factor, return_r=False):
    for i in range(len(params)):
        if not fixed_params[i] is None:
            params[i] = fixed_params[i]

    if type(params) is not np.ndarray:
        params = np.array(params)

    print(f"Params: {', '.join(f'{i:.3f}' for i in params)}", end="\t")
    kon, koff, m, mu, k, c0_1, c0_2, c0_3, c0_4, Vc1, V1, Vc2, V2 = (
        params.tolist() + ctrl_params
    )
    kd = koff / kon
    costs = [0] * 4
    input_params = [c0_1, c0_2, c0_3, c0_4, kon, koff, m, mu, k]
    cost_1 = cost_model(
        params=input_params,
        model=washout_once,
        y0=(c0_1, 0, site_occupancy(c0_1, m, kd)),
        args=(kon, koff, m, mu, k, Vc1, V1),
        t_eval=t_eval[0],
        y_data=y_data[0],
        return_r=return_r,
        media=True,
    )
    cost_2 = cost_model(
        params=input_params,
        model=washout_once_comp,
        y0=(c0_2, 0, site_occupancy(c0_2, m, kd), 0, 10, 0),
        args=(kon, koff, m, mu, k, Vc2, V2),
        t_eval=t_eval[1],
        y_data=y_data[1],
        return_r=return_r,
        media=True,
    )
    # cost_1 = 0
    # cost_2 = 0
    cost_3 = cost_model(
        params=input_params,
        model=washout_zero_bound,
        y0=(c0_3, site_occupancy(c0_3, m, kd)),
        args=(kon, koff, m, mu),
        t_eval=t_eval[2],
        y_data=y_data[2],
        return_r=return_r,
    )
    cost_4 = cost_model(
        params=input_params,
        model=washout_zero_bound,
        y0=(c0_4, site_occupancy(c0_4, m, kd)),
        args=(0, koff, m, mu),
        t_eval=t_eval[3],
        y_data=y_data[3],
        return_r=return_r,
        time_factor=[1, 1, 1, 1, 0.2, 0, 0]
    )
    r = []
    if not return_r:
        costs = [cost_1, cost_2, cost_3, cost_4]
    else:
        costs = [cost_1[0], cost_2[0], cost_3[0], cost_4[0]]
        r = [cost_1[1], cost_2[1], cost_3[1], cost_4[1]]
    # Apply weighting factors and calculate total cost
    weighted_costs = np.array(costs) * factor
    total_cost = weighted_costs.sum()

    # Print results
    print(f"Residue: {', '.join(f'{c:.2f}' for c in costs[:4])}", end="\t")
    print(f"Total Residue: {total_cost:.2f}")

    return (total_cost, r) if return_r else total_cost


def optimize(
    input_file, output_file, tol, bootstrap=False, default_init=True, fit_init=True
):
    y_data_1, y_data_2, t_eval_1 = load_washout_once_data()
    y_data_3, y_data_4, t_eval_2 = load_multi_wash_data()
    if not bootstrap:
        y_data = [y_data_1, y_data_2, y_data_3, y_data_4]
    else:
        y_data = []
        for data in [y_data_1, y_data_2, y_data_3, y_data_4]:
            for j in range(data.shape[1]):
                idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
                data[:, j] = data[idx, j]
            y_data.append(data)
    t_eval = t_eval_1 + t_eval_2
    if input_file == "default":
        model = ModelParams()
        factor = [1, 1, 5, 5]
    else:
        with open(input_file) as f:
            params_dict = yaml.safe_load(f)
        model = ModelParams(**params_dict["MODEL"])
        factor = params_dict["WEIGHTS"]
    # Set initial concentrations
    if default_init:
        model.c0_1 = y_data[0][:, 0].mean()
        model.c0_2 = y_data[1][:, 0].mean()
        model.c0_3 = y_data[2][:, 0].mean()
        model.c0_4 = y_data[3][:, 0].mean()
    c0 = [model.c0_1, model.c0_2, model.c0_3, model.c0_4]
    params = [
        model.kon,
        model.koff,
        model.m,
        model.mu,
        model.k,
    ]
    ctrl_params = c0 + [model.Vc1, model.V1, model.Vc2, model.V2]
    if fit_init:
        params += c0
        ctrl_params = ctrl_params[-4:]
    fixed_params = [None] * len(params)
    params_fit = minimize(
        cost_fn,
        params,
        args=(fixed_params, ctrl_params, y_data, t_eval, factor),
        method="Nelder-Mead",
        tol=tol,
    )
    # Put parameters back into dict
    model_opt = ModelParams(*params_fit.x.tolist())
    results = {"MODEL": model_opt.__dict__, "WEIGHTS": factor}
    # Write parameters to yaml file
    output_file = output_file.replace(".yaml", f"_{str(uuid.uuid4())[:8]}.yaml")
    with open(output_file, "w") as f:
        yaml.dump(results, f)


def main():
    args = docopt(__doc__)
    optimize(
        args["<input_file>"],
        args["<output_file>"],
        float(args["--tol"]),
        args["--bootstrap"],
        args["--default-init"],
        args["--fit-init"],
    )


if __name__ == "__main__":
    main()
