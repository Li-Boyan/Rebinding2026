import numba as nb
import numpy as np
import yaml
from pathlib import Path
from pde import FieldCollection, PDEBase, ScalarField
import h5py
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pde import MemoryStorage, CartesianGrid, ScalarField, FieldCollection, PolarSymGrid


@dataclass
class TissueConfig(object):
    """Parameters describing the tissue"""

    cylindrical: bool = False
    rcap: float = 5
    kon: float = 3.6
    kd: float = 5e-3
    mt: float = 1
    D: float = 4
    c_init: float = 1
    c0: callable = lambda t: 0
    tissue_length: float = 50
    spatial_res: float = 1

    def __post_init__(self):
        self.koff = self.kon * self.kd


@dataclass
class SolverConfig(object):
    """Parameters describing the solver"""

    tmax: float = 100
    dt: float = 0.1
    dt_track: float = 1


class MicroPK(object):
    def __init__(self, tissue_config, solver_config):
        self.tissue_config = tissue_config
        if tissue_config.cylindrical:
            self.grid = PolarSymGrid(
                (self.tissue_config.rcap, self.tissue_config.tissue_length),
                self.tissue_config.tissue_length // self.tissue_config.spatial_res,
            )
        else:
            self.grid = CartesianGrid(
                [[0, self.tissue_config.tissue_length]],
                [self.tissue_config.tissue_length // self.tissue_config.spatial_res],
                periodic=False,
            )
        self.grid_points = self.grid.cell_coords.flatten()
        self.pdeq = DiffusionBindingPDE(tissue_config)
        self.solver_config = solver_config

    def initialize(self, state_data=None):
        if state_data is None:
            c_init = self.tissue_config.c_init
            s_init = c_init / (self.tissue_config.kd + c_init) * self.tissue_config.mt
        else:
            c_init = state_data[0]
            s_init = state_data[1]
        c_state = ScalarField(self.grid, c_init, label="Drug")
        s_state = ScalarField(self.grid, s_init, label="MT occupancy")
        self.init_state = FieldCollection([c_state, s_state])
        self.nvar = len(self.init_state.data)

    def solve(self):
        storage = MemoryStorage()
        final_state = self.pdeq.solve(
            self.init_state,
            t_range=self.solver_config.tmax,
            dt=self.solver_config.dt,
            tracker=["progress", storage.tracker(self.solver_config.dt_track)],
        )
        self.sol = np.array(
            [np.stack([data[i] for data in storage.data]) for i in range(self.nvar)]
        )
        self.t = np.array(storage.times)

    def plot_solutions(
        self, hue="spatial", ax=None, interval=10, cmap=None, log=[True, False]
    ):
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(8, 3))
        elif len(ax) != 2:
            raise ValueError("ax must be a list of length 2")
        if cmap is None:
            cmap = plt.cm.jet
        for j in range(self.nvar):
            if hue == "spatial":
                data = self.sol[j][:, ::interval].T
                x = self.t
            elif hue == "time":
                data = self.sol[j][::interval]
                x = self.grid_points
            cmap_arr = cmap(np.linspace(0, 1, len(data)))
            for i, sol in enumerate(data):
                ax[j].plot(sol, c=cmap_arr[i])
                if log[j]:
                    ax[j].set_yscale("log")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Drug concentration ($\mu M$)")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Binding site occupancy")


def load_settings_to_model(setting_file):
    """Load settings from a YAML file and create a model"""
    with open(setting_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    tissue_config_dict = settings["TISSUE"]
    if tissue_config_dict["c0"][0] == "constant":
        tissue_config_dict["c0"] = float(tissue_config_dict["c0"][1])
    tissue_config = TissueConfig()
    tissue_config = TissueConfig(**tissue_config_dict)
    solver_config = SolverConfig(**settings["SOLVER"])
    micropk = MicroPK(tissue_config, solver_config)
    return micropk


class DiffusionBindingPDE(PDEBase):
    """PDE describing diffusion, binding, and unbinding of a drug"""

    def __init__(self, config):
        """Parameters and boundary conditions"""
        super().__init__()
        self.config = config
        c0 = config.c0
        self.bc = [{"value_expression": config.c0}, {"derivative": 0}]

    def evolution_rate(self, state, t=0):
        """the evolution equation"""
        config = self.config
        c, s = state
        dcdt = (
            config.D * c.laplace(bc=self.bc)
            - config.kon * c * (config.mt - s)
            + config.koff * s
        )
        dsdt = config.kon * c * (config.mt - s) - config.koff * s
        return FieldCollection([dcdt, dsdt])

    def _make_pde_rhs_numba(self, state):
        """the numba-accelerated evolution equation"""
        # make attributes locally available
        config = self.config
        D, kon, kd, koff, mt = (
            config.D,
            config.kon,
            config.kd,
            config.koff,
            config.mt,
        )
        # create operators
        laplace_op = state.grid.make_operator("laplace", bc=self.bc)

        @nb.jit(nopython=True)
        def pde_rhs(state_data, t=0):
            """compiled helper function evaluating right hand side"""
            c = state_data[0]
            s = state_data[1]
            rate = np.empty_like(state_data)
            rate[0] = D * laplace_op(c) - kon * c * (mt - s) + koff * s
            rate[1] = kon * c * (mt - s) - koff * s
            return rate

        return pde_rhs


def run_micropk(
    working_dir: str, load_curr_state: bool = False, output_file: str = "solution.h5"
):
    working_dir = Path(working_dir)
    setting_file = working_dir / "settings.yml"
    save_path = working_dir / output_file
    micropk = load_settings_to_model(setting_file)
    if load_curr_state:
        with h5py.File(working_dir / "solution.h5", "r") as f:
            idx = len([k for k in f.keys() if k.startswith("sol")])
            micropk.initialize(state_data=f[f"sol_{idx-1}"][:, -1, :])
            f.close()
    else:
        micropk.initialize()
    micropk.solve()
    if output_file is None:
        return micropk
    file_access = "a" if load_curr_state else "w"
    with h5py.File(save_path, file_access) as f:
        idx = len([k for k in f.keys() if k.startswith("sol")])
        f.create_dataset(f"sol_{idx}", data=micropk.sol)
        f.create_dataset(f"t_{idx}", data=micropk.t)
    f.close()
