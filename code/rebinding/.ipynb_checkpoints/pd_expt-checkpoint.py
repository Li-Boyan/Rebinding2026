import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def pd_data_processing(raw_data_path, cpds, plate_idx, drop_cpd=None, concs=None):

    raw_data = pd.read_excel(raw_data_path)
    position_info = raw_data.iloc[0].tolist()[2:]
    time = raw_data.Time.tolist()[2:]
    confluency = raw_data.values[2:, 2:].flatten(order="F")

    data = pd.DataFrame()
    data["t"] = time * len(position_info)
    data["t"] //= 60
    data["well"] = np.repeat([s.split(" ")[1] for s in position_info], len(time))
    data["row"] = data.well.apply(lambda s: s[0])
    data["col"] = data.well.apply(lambda s: int(s[1:]))
    data["pos"] = np.repeat([s.split(" ")[-1] for s in position_info], len(time))
    data["confluency"] = confluency
    data["cpd"] = data.row.apply(lambda r: cpds["BCDEFG".find(r) // 2])
    data["plate"] = plate_idx
    data["abs_pos"] = data.apply(lambda df: str(df.plate) + "_" + df.well + "_" + df.pos, axis=1)

    if concs is None:
        concs = np.concatenate([[1e-4], np.logspace(np.log10(0.003), np.log10(30), 9)])
    if type(concs) is np.ndarray:
        concs = [concs] * 3
    cpd_to_concs = {cpd: conc for cpd, conc in zip(cpds, concs)}
    data["c"] = data.apply(lambda df: (cpd_to_concs[df.cpd])[df.col - 2], axis=1)
    data["logc"] = data.c.apply(np.log10)

    if not drop_cpd is None:
        if type(drop_cpd) is str:
            drop_cpd = [drop_cpd]
        for cpd in drop_cpd:
            data.drop(data[data.cpd == cpd].index, inplace=True)
    data.drop(data[data.t > 120].index, inplace=True)

    return data



def plot_cell_growth(cpd, data, axes, cbar_ticks=[1, 3, 5], cbar_ticklabels=[r"$10^{-3}$", r"$10^{-1}$", r"$10^1$"]):
    # _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1, ax2 = tuple(axes)
    data_no_wash = data[np.logical_not(data.Wash) & (data.cpd == cpd)].reset_index(drop=True)
    data_wash = data[(data.Wash) & (data.cpd == cpd)].reset_index(drop=True)
    lp1 = sns.lineplot(
        data=data_no_wash,
        x="t",
        y="confluency",
        hue="lgc",
        ax=ax1,
        legend=False,
        errorbar="se",
        palette="coolwarm",
    )
    lp2 = sns.lineplot(
        data=data_wash,
        x="t",
        y="confluency",
        hue="lgc",
        ax=ax2,
        legend=False,
        errorbar="se",
        palette="coolwarm",
    )
    norm = plt.Normalize(data_wash.lgc.min(), data_wash.lgc.max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    cbaxes = inset_axes(ax1, width="40%", height="5%", loc=2)
    cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_ticklabels, ha="center", fontsize=10)
    cbar.outline.set_edgecolor('none')
    cbar.dividers.set_color('none')


def plot_gr_curves(cpd, data, ax, t_ref=24, line_props=dict(), dose_res_y="gr", colors=["b", "orange"], alpha=1):
    data_no_wash = data[np.logical_not(data.Wash) & (data.cpd == cpd)].reset_index(drop=True)
    data_wash = data[(data.Wash) & (data.cpd == cpd)].reset_index(drop=True)
    t_idx_no_wash = np.argmin(abs(data_no_wash.t - t_ref))
    t_ref_no_wash = data.loc[t_idx_no_wash, "t"]
    t_idx_wash = np.argmin(abs(data_wash.t - t_ref))
    t_ref_wash = data_wash.loc[t_idx_wash, "t"]
    sns.lineplot(
        data=data_no_wash[data_no_wash.t == t_ref_no_wash],
        x="lgc",
        y=dose_res_y,
        color=colors[0],
        **line_props,
        ax=ax,
        legend=False,
        alpha=alpha,
    )
    sns.lineplot(
        data=data_wash[data_wash.t == t_ref_wash],
        x="lgc",
        y=dose_res_y,
        color=colors[1],
        **line_props,
        ax=ax,
        legend=False,
        alpha=alpha,
    )