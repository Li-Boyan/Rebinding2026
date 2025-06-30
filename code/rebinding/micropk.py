import numba as nb
import numpy as np
import yaml
from pathlib import Path
from pde import FieldCollection, PDEBase, ScalarField
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pde import (
    MemoryStorage,
    CartesianGrid,
    ScalarField,
    FieldCollection,
    PolarSymGrid,
    SphericalSymGrid,
)


@dataclass
class TissueConfig(object):
    """Parameters describing the tissue"""

    geometry: str = "1d"
    rcap: float = 5
    variant: str = None
    kon: float = 3.6
    kd: float = 5e-3
    mt: float = 1
    D: float = 4
    s_init: float = 10
    c0: callable = lambda t: 0
    tissue_length: float = 50
    spatial_res: float = 1
    pde: str = None
    r_thresh_power: float = 0.5

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
        if tissue_config.geometry == "cylindrical":
            self.grid = PolarSymGrid(
                (self.tissue_config.rcap, self.tissue_config.tissue_length),
                self.tissue_config.tissue_length // self.tissue_config.spatial_res,
            )
        elif tissue_config.geometry == "1d":
            self.grid = CartesianGrid(
                [[0, self.tissue_config.tissue_length]],
                [self.tissue_config.tissue_length // self.tissue_config.spatial_res],
                periodic=False,
            )
        elif tissue_config.geometry == "spherical":
            self.grid = SphericalSymGrid(
                self.tissue_config.tissue_length,
                self.tissue_config.tissue_length // self.tissue_config.spatial_res,
            )
        self.grid_points = self.grid.cell_coords.flatten()
        if tissue_config.pde == "original":
            self.pdeq = OriginalDiffusionBindingPDE(tissue_config)
        else:
            self.pdeq = DiffusionBindingPDE(tissue_config)
        self.solver_config = solver_config

    def get_c_init(self):
        s_init = self.tissue_config.s_init
        return s_init + self.tissue_config.kd * s_init / (
            self.tissue_config.mt - s_init
        )

    def initialize(self, state_data=None):
        """Set the initial condition of the system"""
        if state_data is None:
            s_init = self.tissue_config.s_init
            c_init = self.get_c_init()
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
            solver="scipy",
        )
        self.sol = np.array(
            [np.stack([data[i] for data in storage.data]) for i in range(self.nvar)]
        )
        self.t = np.array(storage.times)

    def plot_curves(
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
                x = self.t / 3600
            elif hue == "time":
                data = self.sol[j][::interval]
                x = self.grid_points
            cmap_arr = cmap(np.linspace(0, 1, len(data)))
            if j == 1:
                data /= self.tissue_config.mt
            for i, sol in enumerate(data):
                ax[j].plot(x, sol, c=cmap_arr[i])
                if log[j]:
                    ax[j].set_yscale("log")
        xlabel = "Time (h)" if hue == "spatial" else "Position ($\mu m$)"
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel("Drug concentration ($\mu M$)")
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel("Binding site occupancy")
        return ax

    def plot_kymograph(
        self,
        ax=None,
        s_thresh=0.5,
        tspan=None,
        ytickinterval=1440,
        ylabel="Time (h)",
        vmin_c=0,
        vmax_c=None,
    ):
        c_thresh = self.tissue_config.mt * (
            s_thresh
            + self.tissue_config.kd * s_thresh / (self.tissue_config.mt - s_thresh)
        )
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(12, 3))
        plt.subplots_adjust(wspace=0.2)
        tspan = len(self.t) if tspan is None else int(tspan)
        if vmax_c is None:
            vmax_c = self.get_c_init()
        if self.tissue_config.pde == "original":
            sns.heatmap(
                self.sol[:, 0 : tspan + 1, :].sum(axis=0),
                ax=ax[0],
                cmap="seismic",
                center=c_thresh,
                vmin=vmin_c,
                vmax=vmax_c,
            )
        else:
            sns.heatmap(
                self.sol[0, 0 : tspan + 1, :],
                ax=ax[0],
                center=c_thresh,
                cmap="seismic",
                vmin=vmin_c,
                vmax=vmax_c,
            )
        sns.heatmap(
            self.sol[1, 0 : tspan + 1, :] / self.tissue_config.mt,
            ax=ax[1],
            center=s_thresh,
            cmap="seismic",
            vmin=0,
            vmax=1.0,
        )
        for a in ax:
            a.set_xlabel("Position ($\mu m$)")
            a.set_ylabel(ylabel)
            a.set_yticks(np.arange(0, tspan + 1, ytickinterval))
            a.set_yticklabels(
                (
                    np.arange(0, tspan + 1, ytickinterval)
                    // self.solver_config.dt_track
                ).astype(int)
            )
            a.set_xticks(np.arange(0, self.tissue_config.tissue_length + 1, 10))
            a.set_xticklabels(np.arange(0, self.tissue_config.tissue_length + 1, 10))

        return ax

    def decay_time(self, idx, thresh):
        if idx == 1:
            thresh *= self.tissue_config.mt
        return np.where(np.diff(self.sol[idx] < thresh, axis=0))[0]


def load_settings_to_model(setting_file):
    """Load settings from a YAML file and create a model"""
    with open(setting_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    tissue_config_dict = settings["TISSUE"]
    if tissue_config_dict["c0"][0] == "constant":
        tissue_config_dict["c0"] = float(tissue_config_dict["c0"][1])
    tissue_config = TissueConfig(**tissue_config_dict)
    solver_config = SolverConfig(**settings["SOLVER"])
    micropk = MicroPK(tissue_config, solver_config)
    return micropk


def load_sol_to_model(working_dir):
    """Load solution from a HDF5 file and create a model"""
    if working_dir is str:
        working_dir = Path(working_dir)
    solution_file = working_dir / "solution.h5"
    setting_file = working_dir / "settings.yml"
    micropk = load_settings_to_model(setting_file)
    micropk.initialize()
    f = h5py.File(solution_file)
    nperiod = len([k for k in f.keys() if k.startswith("sol")])
    micropk.sol = np.concatenate([f[f"sol_{i}"][:] for i in range(nperiod)], axis=1)
    tend = [0] + np.cumsum([f[f"t_{i}"][-1] for i in range(nperiod)][:-1]).tolist()
    micropk.t = np.concatenate([f[f"t_{i}"][:] + tend[i] for i in range(nperiod)])
    f.close()
    return micropk


class DiffusionBindingPDE(PDEBase):
    """PDE describing diffusion, binding, and unbinding of a drug"""

    def __init__(self, config):
        """Parameters and boundary conditions"""
        super().__init__()
        self.check_implementation = False
        self.config = config
        c0 = config.c0
        if config.geometry == "1d" or config.geometry == "cylindrical":
            self.bc = [{"value_expression": c0}, {"derivative": 0}]
        elif config.geometry == "spherical":
            self.bc = [{"value_expression": c0}]

    def evolution_rate(self, state, t=0):
        """the evolution equation"""
        config = self.config
        c, s = state
        dcdt = config.D * ((c - s).laplace(bc=self.bc))
        dsdt = config.kon * (c - s) * (config.mt - s) - config.koff * s
        return FieldCollection([dcdt, dsdt])

    def _make_pde_rhs_numba(self, state):
        """the numba-accelerated evolution equation"""
        # make attributes locally available
        config = self.config
        D, kon, kd, koff, mt, var, r_thresh_power = (
            config.D,
            config.kon,
            config.kd,
            config.koff,
            config.mt,
            config.variant,
            config.r_thresh_power,
        )
        # create operators
        laplace_op = state.grid.make_operator("laplace", bc=self.bc)
        d1 = state.grid.make_operator("d_dr", bc=self.bc)
        d2 = state.grid.make_operator("d2_dr2", bc=self.bc)

        @nb.jit(nopython=True)
        def pde_rhs(state_data, t=0):
            """compiled helper function evaluating right hand side"""
            c = state_data[0]
            s = state_data[1]
            rate = np.empty_like(state_data)
            f = np.empty_like(c)
            if var is None:
                rate[0] = D * laplace_op(c - s)
            elif var == "piecewise_approx":
                r = kd / mt
                r_thresh = r**r_thresh_power
                c_thresh = mt * (1 - r_thresh)
                s_thresh = mt * (1 - r_thresh * 2 + r)
                for i in range(c.shape[0]):
                    if c[i] >= c_thresh:
                        f[i] = c[i] - s_thresh
                    else:
                        denom = 1.0 - c[i] / mt
                        f[i] = r * c[i] / denom
                rate[0] =  D * laplace_op(f)
            rate[1] = kon * (c - s) * (mt - s) - koff * s
            return rate

        return pde_rhs


class OriginalDiffusionBindingPDE(PDEBase):
    """PDE describing diffusion, binding, and unbinding of a drug"""

    def __init__(self, config):
        """Parameters and boundary conditions"""
        super().__init__()
        self.check_implementation = False
        self.config = config
        c0 = config.c0
        if config.geometry == "1d" or config.geometry == "cylindrical":
            self.bc = [{"value_expression": c0}, {"derivative": 0}]
        elif config.geometry == "spherical":
            self.bc = [{"value_expression": c0}]

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
        D, kon, kd, koff, mt, var = (
            config.D,
            config.kon,
            config.kd,
            config.koff,
            config.mt,
            config.variant,
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


def major_eigv_cylindrical_diffusion(r1, r2):
    r1 = config.rcap
    r2 = config.tissue_length
    lamda = np.linspace(1e-4, 0.005, 100)
    f1 = jv(0, lamda**0.5 * r1) * yv(1, lamda**0.5 * r2)
    f2 = jv(1, lamda**0.5 * r2) * yv(0, lamda**0.5 * r1)
    eigv = lamda[np.argmin(abs(f1 - f2))]
    return eigv
