import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tomllib as tl
from scipy.special import expit as sigmoid
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

DATA_DIR = "/cmnfs/data/cell_viability/CTRP/v2/curvecurator_per_drug/"
ANNOTATION_FILE = "/cmnfs/proj/sysbiomed24/model_list_20240110.csv"

def to_p_space(x: np.ndarray):
    x = np.where(x == 0, 1e-9, x) # Avoid log(0)
    return -np.log10(x)

def from_p_space(x: np.ndarray):
    return 10 ** -x

def load_curvecurator(drug_id):
    with open(f"{DATA_DIR}{drug_id}/config.toml", "rb") as f:
        toml = tl.load(f)['Experiment']

    df_curvecurator = pd.read_csv(f"{DATA_DIR}{drug_id}/curves.txt", sep="\t", index_col=0)
    
    experiments = toml['experiments']
    doses = toml['doses']

    df_doses = pd.DataFrame(doses)
    df_doses.index = [f"Raw {experiment}" for experiment in experiments]
    df_doses.columns = ["Dose"]
    df_doses["Dose"] = df_doses["Dose"] / 1e6 # Convert from µM to M
    df_doses["pDose"] = to_p_space(df_doses["Dose"])

    df_annotation = pd.read_csv("/cmnfs/proj/sysbiomed24/model_list_20240110.csv")
    df_annotation = df_annotation[df_annotation["RRID"].notna()]
    df_annotation.index = df_annotation["RRID"]
    index_intersection = df_curvecurator.index.intersection(df_annotation.index)
    df_annotation = df_annotation.loc[index_intersection]
    df_curvecurator = df_curvecurator.loc[index_intersection]
    df_curvecurator = pd.concat([df_curvecurator, df_annotation], axis=1)

    return df_curvecurator, df_doses

def __logistic_decay__(x, front, back, slope, ec50):
    return (front - back) * sigmoid(slope * (x - ec50))

def __get_actual_parameters__(front, mid_ratio, back_ratio, slope_1, slope_2, pec50_1, pec50_delta):
    middle = front * mid_ratio
    back = middle * back_ratio

    pec50_2 = pec50_1 + pec50_delta

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

def __fit_double_logistic__(df_doses: pd.DataFrame, cc_cell_line: pd.Series):
    df_intensitites = df_doses.join(cc_cell_line)
    df_intensitites = df_intensitites.dropna()
    df_intensitites.columns = df_doses.columns.tolist() + ["Intensity"]

    X = df_intensitites["pDose"].to_numpy()
    Y = df_intensitites["Intensity"].to_numpy()

    MIN = 0
    INIT = 1
    MAX = 2

    constraints = {
        "front": (0.1, 1, 1.5),
        "mid_ratio": (0, 0.5, 1),
        "back_ratio": (0, 0.5, 1),
        "slope": (0, 1, 20),
        "pec50_1": (4, 5, 9),
        "pec50_delta": (1, 3, 8),
    }
    param_order = ["front", "mid_ratio", "back_ratio", "slope", "slope", "pec50_1", "pec50_delta"]

    popt, _ = curve_fit(__double_logistic__, X, Y,
                        maxfev=int(1e6),
                        p0=[constraints[param][INIT] for param in param_order],
                        method='trf',
                        bounds=(
                            [constraints[param][MIN] for param in param_order],
                            [constraints[param][MAX] for param in param_order],
                        ))
    r2 = r2_score(Y, __double_logistic__(X, *popt))

    return r2, popt

def fit(df_curvecurator: pd.DataFrame, df_doses: pd.DataFrame, target_pec50_range: np.ndarray):
    df_curvecurator["single_r2"] = df_curvecurator["Curve R2"]
    fit_res = df_curvecurator.apply(lambda x: __fit_double_logistic__(df_doses, x), axis=1)
    df_curvecurator["double_r2"] = fit_res.apply(lambda x: x[0])
    df_curvecurator["double_params"] = fit_res.apply(lambda x: x[1])
    df_curvecurator["single_params"] = df_curvecurator.apply(lambda x: [x["Curve Front"], x["Curve Back"], x["Curve Slope"] * 2, x["pEC50"]], axis=1)

    df_curvecurator["sigmoid_diff"] = df_curvecurator["double_r2"] - df_curvecurator["single_r2"]

    target_range_start = target_pec50_range[0]
    target_range_end = target_pec50_range[1]
    df_curvecurator["double_target_effect_size"] = df_curvecurator.apply(lambda x: __double_logistic__(target_range_start, *x["double_params"]) - __double_logistic__(target_range_end, *x["double_params"]), axis=1)
    df_curvecurator["double_global_effect_size"] = df_curvecurator.apply(lambda x: __double_logistic__(-np.inf, *x["double_params"]) - __double_logistic__(np.inf, *x["double_params"]), axis=1)
    df_curvecurator["single_target_effect_size"] = df_curvecurator.apply(lambda x: __single_logistic__(target_range_start, *x["single_params"]) - __single_logistic__(target_range_end, *x["single_params"]), axis=1)
    df_curvecurator["single_global_effect_size"] = df_curvecurator.apply(lambda x: __single_logistic__(-np.inf, *x["single_params"]) - __single_logistic__(np.inf, *x["single_params"]), axis=1)

    def get_fit_type(row, fit: str):
        target_percentage = row[f"{fit}_target_effect_size"] / row[f"{fit}_global_effect_size"]
        if target_percentage > 0.1:
            if target_percentage < 0.9:
                return "Both"
            else:
                return "Target"
        else:
            return "Off-target"

    df_curvecurator["Double fit type"] = df_curvecurator.apply(lambda x: get_fit_type(x, "double"), axis=1)
    df_curvecurator["Single fit type"] = df_curvecurator.apply(lambda x: get_fit_type(x, "single"), axis=1)

    return df_curvecurator

def plot_fit(df_curvecurator: pd.DataFrame, df_doses: pd.DataFrame, cell_line: str, target_ec50_range: np.ndarray = None):
    cc_cell_line = df_curvecurator.T[cell_line]
    X = np.linspace(df_doses["pDose"].min(), df_doses["pDose"].max(), 1000)
    Y_single = __single_logistic__(X, *cc_cell_line["single_params"])
    Y_double = __double_logistic__(X, *cc_cell_line["double_params"])

    df_intensities = df_doses.join(df_curvecurator.T[cell_line])
    df_intensities.columns = df_doses.columns.tolist() + ["Intensity"]

    X = from_p_space(X)

    sns.scatterplot(data=df_intensities, x="Dose", y="Intensity")
    plt.plot(X, Y_single, color="red")
    plt.plot(X, Y_double, color="green")

    if target_ec50_range is not None:
        plt.axvline(target_ec50_range[0], color="black", linestyle="--")
        plt.axvline(target_ec50_range[1], color="black", linestyle="--")

    plt.title(cell_line)
    plt.xscale("log")
    plt.xlabel("Dose (M)")
    plt.ylabel("Relative intensity")

def plot_fit_type(df_curvecurator: pd.DataFrame, category: str, drug: str = None, palette: dict = None):
    df_plot = df_curvecurator[[category, "Fit type"]].groupby([category, "Fit type"]).size().unstack().fillna(0)
    df_plot.sort_values("Both", ascending=False, inplace=True)
    df_plot.plot.bar(stacked=True, color=[palette[col] for col in df_plot.columns])

    # Set size dynamically
    plt.gcf().set_size_inches(0.2 * len(df_plot) + 2, 5)

    plt.ylabel("Frequency")
    plt.xlabel("Cancer type")
    plt.title(f"{drug} - Fit type by {category.replace('_', ' ')}", fontsize=15)
