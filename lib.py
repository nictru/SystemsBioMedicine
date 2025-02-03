import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tomllib as tl
from scipy.special import expit as sigmoid
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

DATA_DIR = "/cmnfs/data/cell_viability/CTRP/v2/curvecurator_per_drug/"
ANNOTATION_FILE = "/cmnfs/proj/sysbiomed24/model_list_20240110.csv"
X_MIN=0.002
X_MAX=66

def load_curvecurator(drug_id):
    with open(f"{DATA_DIR}{drug_id}/config.toml", "rb") as f:
        toml = tl.load(f)['Experiment']

    df_curvecurator = pd.read_csv(f"{DATA_DIR}{drug_id}/curves.txt", sep="\t", index_col=0)
    
    experiments = toml['experiments']
    doses = toml['doses']

    df_doses = pd.DataFrame(doses)
    df_doses.index = [f"Raw {experiment}" for experiment in experiments]
    df_doses.columns = ["Dose"]

    df_annotation = pd.read_csv("../../model_list_20240110.csv")
    df_annotation = df_annotation[df_annotation["RRID"].notna()]
    df_annotation.index = df_annotation["RRID"]
    index_intersection = df_curvecurator.index.intersection(df_annotation.index)
    df_annotation = df_annotation.loc[index_intersection]
    df_curvecurator = df_curvecurator.loc[index_intersection]
    df_curvecurator = pd.concat([df_curvecurator, df_annotation], axis=1)
    df_curvecurator["EC50"] = 1e6 * (10 ** -df_curvecurator["pEC50"])

    # Remove rows from df_doses, where the corresponding column in curvecurator has only one value
    # This mainly targets the "Raw 0" and "Raw 1" columns, as they are always 1
    # By removing these columns, we allow the curve to fit the "interesting" part of the curve
    unique_counts = df_curvecurator[df_doses.index].nunique()
    unique_counts = unique_counts[unique_counts == 1]
    singular_columns = unique_counts.index.to_list()
    df_doses = df_doses.drop(singular_columns, axis=0)

    return df_curvecurator, df_doses

def __logistic_decay__(x, front, back, slope, ec50):
    return (front - back) * sigmoid(- slope * (x - ec50))

def __get_actual_parameters__(front, mid_ratio, back_ratio, slope_1, slope_2, pec50_1, pec50_delta):
    middle = front * mid_ratio
    back = middle * back_ratio

    pec50_2 = pec50_1 + 10 ** pec50_delta

    slope_1 = slope_1 / pec50_1
    slope_2 = slope_2 / pec50_2
    
    return front, middle, back, slope_1, slope_2, pec50_1, pec50_2

# Define the double logistic function (piecewise)
# mid_ratio and back_ratio are defined relative to front
# This is to ensure that the function is always decreasing: We can easily set bounds for the optimization by setting the ratios to be between 0 and 1
# With absolute values it would be harder to ensure that the function is always decreasing
# Similarly for the pec50, here the delta is even defined in log space (not sure if this changes much)
def __double_logistic__(x, front, mid_ratio, back_ratio, slope1, slope2, pec50_1, pec50_delta):
    front, middle, back, slope1, slope2, pec50_1, pec50_2 = __get_actual_parameters__(front, mid_ratio, back_ratio, slope1, slope2, pec50_1, pec50_delta)

    # First logistic function (decays from 1 to b)
    phase1 = __logistic_decay__(x, front, middle, slope1, pec50_1)
    # Second logistic function (decays from b to 0)
    phase2 = __logistic_decay__(x, middle, back, slope2, pec50_2)

    # Combine both phases
    return phase1 + phase2 + back

def __single_logistic__(x, front, back, slope, ec50):
    return __logistic_decay__(x, front, back, slope, ec50) + back

def plot_fit(df_curvecurator: pd.DataFrame, df_doses: pd.DataFrame, cell_line: str, target_range_start: float= None, target_range_end: float = None, title: str = None, technical: bool = False, single: bool = True):
    pred_fun = __double_logistic__

    x = df_doses["Dose"]
    y = df_curvecurator[df_doses.index].loc[cell_line]

    r2 = df_curvecurator.loc[cell_line]["double_r2"]
    params = df_curvecurator.loc[cell_line]["double_params"]

    front, mid_ratio, back_ratio, slope1, slope2, pec50_1, pec50_delta = params
    front, middle, back, slope1, slope2, pec50_1, pec50_2 = __get_actual_parameters__(front, mid_ratio, back_ratio, slope1, slope2, pec50_1, pec50_delta)

    target_effect = df_curvecurator.loc[cell_line]["target_effect_size"]

    plt.scatter(x, y, label='Data', color='blue')
    x_gen = np.logspace(np.log10(X_MIN), np.log10(x.max()), 10000)
    plt.plot(x_gen, pred_fun(x_gen, *params), label='Fitted Model', color='red')

    if target_range_start and target_range_end:
        plt.axvline(target_range_start, color='black', linestyle='--', label='Target range start')
        plt.axvline(target_range_end, color='black', linestyle='--', label='Target range end')

    if single:
        front = df_curvecurator.loc[cell_line]["Curve Front"]
        back = df_curvecurator.loc[cell_line]["Curve Back"]
        slope = df_curvecurator.loc[cell_line]["Curve Slope"]
        slope = slope / df_curvecurator.loc[cell_line]["EC50"]
        ec50 = df_curvecurator.loc[cell_line]["EC50"]
        plt.axvline(ec50, color='green', linestyle='--', label='EC50')
        plt.plot(x_gen, __single_logistic__(x_gen, front, back, slope, ec50), linestyle='--', label='Single logistic', color='green')
        print(f"Front: {front:.2f}, Back: {back:.2f}, Slope: {slope}, EC50: {ec50:.2f}")

    if technical:
        x_center = 10 ** ((np.log10(pec50_1) + np.log10(pec50_2)) / 2)

        y_max = pred_fun(0, *params)
        y_min = pred_fun(np.inf, *params)
        y_center = pred_fun(x_center, *params)

        # Plot first logistic function
        plt.plot(x_gen, __logistic_decay__(x_gen, front, middle, slope1, pec50_1), linestyle='--', label='Front sigmoid', color='green')
        # Plot second logistic function
        plt.plot(x_gen, __logistic_decay__(x_gen, middle, back, slope2, pec50_2), linestyle='--', label='Back sigmoid', color='purple')

        # Show vertical lines for pec50
        plt.axvline(pec50_1, color='green', linestyle='-.', label='PEC50 1')
        plt.axvline(pec50_2, color='green', linestyle='-.', label='PEC50 2')
        plt.axvline(x_center, color='purple', linestyle='-.', label='Center')
        # Show horizontal lines for min, max, center
        plt.axhline(y_min, color='black', linestyle='-.', label='Min')
        plt.axhline(y_max, color='black', linestyle='-.', label='Max')
        plt.axhline(y_center, color='black', linestyle='-.', label='Center')

    # Remove frame
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # x log scale
    plt.xscale('log')
    plt.legend()
    plt.title(f'{title or cell_line}\nR2: {r2:.2f}\nTarget effect: {target_effect:.2f}')
    # Set axis titles
    plt.xlabel("Concentration in µM")
    plt.ylabel("Relative intensity")
    plt.show()
    plt.close()
    print(f"Front: {front:.2f}, Middle: {middle:.2f}, Back: {back:.2f}, PEC50 1: {pec50_1:.2f}, PEC50 2: {pec50_2:.2f}, Slope 1: {slope1:.2f}, Slope 2: {slope2:.2f}")

def __fit_double_logistic__(doses: pd.Series, curvecurator: pd.Series):
    x = doses["Dose"]
    y = curvecurator[doses.index]
    mask = y.notna()
    x = x[mask].values
    y = y[mask].values

    initial_ec50_1 = max(0.003, curvecurator["EC50"] / 10)

    MIN = 0
    INIT = 1
    MAX = 2

    constraints = {
        "front": (0.1, max(0.11, curvecurator["Curve Front"]), 1.5),
        "mid_ratio": (0, 0.5, 1),
        "back_ratio": (0, 0.5, 1),
        "slope": (0, 1, 100),
        "ec50_1": (0, initial_ec50_1, X_MAX),
        "ec50_delta": (0, 2, 5),
    }
    param_order = ["front", "mid_ratio", "back_ratio", "slope", "slope", "ec50_1", "ec50_delta"]

    popt, _ = curve_fit(__double_logistic__, x, y,
                        maxfev=int(1e6),
                        p0=[constraints[param][INIT] for param in param_order],
                        method='trf',
                        bounds=(
                            [constraints[param][MIN] for param in param_order],
                            [constraints[param][MAX] for param in param_order],
                        ))
    r2 = r2_score(y, __double_logistic__(x, *popt))

    return r2, popt

def fit(df_curvecurator: pd.DataFrame, df_doses: pd.DataFrame, target_range_start:float, target_range_end:float):
    df_curvecurator["single_r2"] = df_curvecurator["Curve R2"]
    df_curvecurator["double_fit_res"] = df_curvecurator.apply(lambda x: __fit_double_logistic__(df_doses, x), axis=1)
    df_curvecurator["double_r2"] = df_curvecurator["double_fit_res"].apply(lambda x: x[0])
    df_curvecurator["double_params"] = df_curvecurator["double_fit_res"].apply(lambda x: x[1])
    df_curvecurator["double_pec50_1"] = df_curvecurator["double_params"].apply(lambda x: x[5])
    df_curvecurator["double_pec50_2"] = df_curvecurator["double_params"].apply(lambda x: x[5] + 10 ** x[6])
    df_curvecurator["double_x_center"] = df_curvecurator.apply(lambda x: 10 ** ((np.log10(x["double_pec50_1"]) + np.log10(x["double_pec50_2"])) / 2), axis=1)
    df_curvecurator["double_y_max"] = df_curvecurator.apply(lambda x: __double_logistic__(0, *x["double_params"]), axis=1)
    df_curvecurator["double_y_center"] = df_curvecurator.apply(lambda x: __double_logistic__(x["double_x_center"], *x["double_params"]), axis=1)

    df_curvecurator["Global effect"] = 2 ** df_curvecurator["Curve Fold Change"]
    df_curvecurator["sigmoid_diff"] = df_curvecurator["double_r2"] - df_curvecurator["single_r2"]
    df_curvecurator["single_step_is_target"] = (target_range_start < (1e3 * df_curvecurator["EC50"])) & (target_range_end > (1e3 * df_curvecurator["EC50"]))

    df_curvecurator["target_effect_size"] = df_curvecurator.apply(lambda x: __double_logistic__(target_range_start, *x["double_params"]) - __double_logistic__(target_range_end, *x["double_params"]), axis=1)
    df_curvecurator["two_step_global_effect_size"] = df_curvecurator.apply(lambda x: __double_logistic__(0, *x["double_params"]) - __double_logistic__(X_MAX, *x["double_params"]), axis=1)
    
    def get_fit_type(row):
        target_percentage = row["target_effect_size"] / row["two_step_global_effect_size"]
        if target_percentage > 0.1:
            if target_percentage < 0.9:
                return "Both"
            else:
                return "Target"
        else:
            return "Off-target"

    df_curvecurator["Fit type"] = df_curvecurator.apply(get_fit_type, axis=1)

    return df_curvecurator

def plot_fit_type(df_curvecurator: pd.DataFrame, category: str, drug: str = None, palette: dict = None):
    df_plot = df_curvecurator[[category, "Fit type"]].groupby([category, "Fit type"]).size().unstack().fillna(0)
    df_plot.sort_values("Both", ascending=False, inplace=True)
    df_plot.plot.bar(stacked=True, color=[palette[col] for col in df_plot.columns])

    # Set size dynamically
    plt.gcf().set_size_inches(0.2 * len(df_plot) + 2, 5)

    plt.ylabel("Frequency")
    plt.xlabel("Cancer type")
    plt.title(f"{drug} - Fit type by {category.replace('_', ' ')}", fontsize=15)
