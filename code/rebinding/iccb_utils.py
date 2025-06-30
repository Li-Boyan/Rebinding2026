import pandas as pd
import numpy as np


def load_iccb_ixm_puncta_data(
    path,
    cell_count_col="Cell Count (bol816 blob detection)",
    puncta_cols=[
        "PolyIC feature count.Total (bol816 blob detection)",
        "SG feature count.Total (bol816 blob detection)",
    ],
):
    # Load data
    data = pd.read_csv(path, sep="\t")
    # Drop duplicate columns
    data.drop(columns=[col for col in data.columns if col.endswith(".1")], inplace=True)
    # Remove site wo cells
    site_data = data[np.logical_not(pd.isna(data[cell_count_col]))].reset_index(
        drop=True
    )
    well_data = site_data.groupby("Well Name")[[cell_count_col] + puncta_cols].sum()
    well_data.columns = ["nCell"] + ["puncta_%d" % i for i in range(len(puncta_cols))]
    for i in range(len(puncta_cols)):
        well_data["puncta_%d_per_cell" % i] = (
            well_data["puncta_%d" % i] / well_data["nCell"]
        )
    return data, site_data, well_data
