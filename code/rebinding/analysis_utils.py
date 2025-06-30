import string
import numpy as np

iccb_col_rename_dict = {
    "Plate ID": "plate",
    "Well Name": "well",
    "Site ID": "site",
    "MEASUREMENT SET ID": "measurement",
    "Cell Count (ClarenceKRDspot detection)": "nCell",
}


def group_number(well, start_row, start_col, nrows_per_group, ncols_per_group, nGroups_by_col):
    row = string.ascii_uppercase.find(well[0])
    col = int(well[1:])
    row_group_num = (row - start_row) // nrows_per_group
    col_group_num = (col - start_col) // ncols_per_group
    return row_group_num * nGroups_by_col + col_group_num


def rename_columns(data, nfeature_col):
    rename_dict = iccb_col_rename_dict.update({nfeature_col: "nFeatures"})
    return [iccb_col_rename_dict.get(col) for col in data.columns]

def dropwells(data, wells_to_drop):
    find_wells = np.array([False] * len(data))
    for well in wells_to_drop:
        find_wells |= (data.group == well)
    return data.drop(np.where(find_wells)[0])
    
