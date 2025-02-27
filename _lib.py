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
    """
    Convert from dose to pDose
    """
    x = np.where(x == 0, 1e-10, x) # Avoid log(0)
    return -np.log10(x)

def from_p_space(x: np.ndarray):
    """
    Convert from pDose to dose
    """
    return 10 ** -x

def load_curvecurator(drug_id):
    """
    Load the curvecurator data for a given drug.
    Only keeps the cell lines that are annotated.
    """
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

    df_annotation = pd.read_csv(ANNOTATION_FILE)
    df_annotation = df_annotation[df_annotation["RRID"].notna()]
    df_annotation.index = df_annotation["RRID"]
    index_intersection = df_curvecurator.index.intersection(df_annotation.index)
    df_annotation = df_annotation.loc[index_intersection]
    df_curvecurator = df_curvecurator.loc[index_intersection]
    df_curvecurator = pd.concat([df_curvecurator, df_annotation], axis=1)

    return df_curvecurator, df_doses

def __logistic_decay__(x: np.ndarray, front: float, back: float, slope: float, ec50: float):
    """
    Logistic decay function with additional parameters.

    Parameters
    ----------
    x: The x-values
    front: The front of the logistic decay
    back: The back of the logistic decay
    slope: The slope of the logistic decay
    ec50: The ec50 of the logistic decay

    Returns
    -------
    np.ndarray: The y-values
    """
    return (front - back) * sigmoid(slope * (x - ec50))

def __get_actual_parameters__(front: float, mid_ratio: float, back_ratio: float, slope_1: float, slope_2: float, pec50_1: float, pec50_delta: float):
    """
    Get the actual parameters from the relative parameters.
    """

    middle = front * mid_ratio
    back = middle * back_ratio

    pec50_2 = pec50_1 + pec50_delta

    return front, middle, back, slope_1, slope_2, pec50_1, pec50_2

def __double_logistic__(x: np.ndarray, front: float, mid_ratio: float, back_ratio: float, slope1: float, slope2: float, pec50_1: float, pec50_delta: float):
    """
    Double logistic function with additional parameters.
    The relative parameters are used because this way the bounds can be set easily.

    Parameters
    ----------
    x: The x-values
    front: The front of the logistic decay
    mid_ratio: The ratio of the middle to the front
    back_ratio: The ratio of the back to the middle
    slope1: The slope of the first logistic decay
    slope2: The slope of the second logistic decay
    pec50_1: The pEC50 of the first logistic decay
    pec50_delta: The difference in pEC50 between the two logistic decays
    """

    front, middle, back, slope1, slope2, pec50_1, pec50_2 = __get_actual_parameters__(front, mid_ratio, back_ratio, slope1, slope2, pec50_1, pec50_delta)

    logistic_1 = __logistic_decay__(x, front, middle, slope1, pec50_1)
    logistic_2 = __logistic_decay__(x, middle, back, slope2, pec50_2)

    return logistic_1 + logistic_2 + back

def __single_logistic__(x: np.ndarray, front: float, back: float, slope: float, ec50: float):
    """
    Single logistic function.
    Adds an offset to the logistic decay.
    """
    return __logistic_decay__(x, front, back, slope, ec50) + back

def __fit_double_logistic__(df_doses: pd.DataFrame, cc_cell_line: pd.Series):
    """
    Fit the double logistic function to a single cell line.
    """
    df_intensitites = df_doses.join(cc_cell_line) # We need the actual doses as X values
    df_intensitites = df_intensitites.dropna() # NaN values lead to errors
    df_intensitites.columns = df_doses.columns.tolist() + ["Intensity"]

    X = df_intensitites["pDose"].to_numpy()
    Y = df_intensitites["Intensity"].to_numpy()

    # This constraints led to the best results
    MIN, INIT, MAX = 0, 1, 2

    constraints = {
        "front": (0.1, 1, 1.5),
        "mid_ratio": (0, 0.5, 1),
        "back_ratio": (0, 0.5, 1),
        "slope": (0, 1, 20),
        "pec50_1": (4, 5, 9),
        "pec50_delta": (0, 3, 5),
    }
    param_order = ["front", "mid_ratio", "back_ratio", "slope", "slope", "pec50_1", "pec50_delta"]

    popt, _ = curve_fit(__double_logistic__, X, Y,
                        maxfev=int(1e6), # Sometimes the model does not converge with the default maxfev
                        p0=[constraints[param][INIT] for param in param_order], # Initial guess
                        method='trf', # Recommended when using bounds
                        bounds=(
                            [constraints[param][MIN] for param in param_order], # Lower bounds
                            [constraints[param][MAX] for param in param_order], # Upper bounds
                        ))
    r2 = r2_score(Y, __double_logistic__(X, *popt))

    return r2, popt

def fit(df_curvecurator: pd.DataFrame, df_doses: pd.DataFrame, target_pec50_range: np.ndarray):
    """
    Fits the double logistic function to all cell lines and calculates metrics.
    """
    df_curvecurator["single_r2"] = df_curvecurator["Curve R2"]
    df_curvecurator["single_params"] = df_curvecurator.apply(lambda x: [x["Curve Front"], x["Curve Back"], x["Curve Slope"] * 2, x["pEC50"]], axis=1)

    fit_res = df_curvecurator.apply(lambda x: __fit_double_logistic__(df_doses, x), axis=1)
    df_curvecurator["double_r2"] = fit_res.apply(lambda x: x[0])
    df_curvecurator["double_params"] = fit_res.apply(lambda x: x[1])
    df_curvecurator["double_front"] = df_curvecurator["double_params"].map(lambda x: x[0])
    df_curvecurator["double_plateau"] = df_curvecurator["double_params"].map(lambda x: x[0] * x[1])
    df_curvecurator["double_back"] = df_curvecurator["double_params"].map(lambda x: x[0] * x[1] * x[2])
    df_curvecurator["double_pec50_1"] = df_curvecurator["double_params"].map(lambda x: x[5])
    df_curvecurator["double_pec50_1_stepsize"] = df_curvecurator.apply(lambda x: x["double_front"] - x["double_plateau"], axis=1)
    df_curvecurator["double_pec50_2"] = df_curvecurator["double_params"].map(lambda x: x[5] + x[6])
    df_curvecurator["double_pec50_2_stepsize"] = df_curvecurator.apply(lambda x: x["double_plateau"] - x["double_back"], axis=1)

    df_curvecurator["sigmoid_diff"] = df_curvecurator["double_r2"] - df_curvecurator["single_r2"]

    target_range_start = target_pec50_range[0]
    target_range_end = target_pec50_range[1]
    df_curvecurator["double_target_effect_size"] = df_curvecurator.apply(lambda x: __double_logistic__(target_range_end, *x["double_params"]) - __double_logistic__(target_range_start, *x["double_params"]), axis=1)
    df_curvecurator["double_global_effect_size"] = df_curvecurator.apply(lambda x: __double_logistic__(np.inf, *x["double_params"]) - __double_logistic__(-np.inf, *x["double_params"]), axis=1)
    df_curvecurator["single_target_effect_size"] = df_curvecurator.apply(lambda x: __single_logistic__(target_range_end, *x["single_params"]) - __single_logistic__(target_range_start, *x["single_params"]), axis=1)
    df_curvecurator["single_global_effect_size"] = df_curvecurator.apply(lambda x: __single_logistic__(np.inf, *x["single_params"]) - __single_logistic__(-np.inf, *x["single_params"]), axis=1)

    df_curvecurator["double_pec50_1_substantial"] = df_curvecurator["double_pec50_1_stepsize"] / df_curvecurator["double_global_effect_size"] > 0.1
    df_curvecurator["double_pec50_2_substantial"] = df_curvecurator["double_pec50_2_stepsize"] / df_curvecurator["double_global_effect_size"] > 0.1

    df_curvecurator["Effect span"] = df_curvecurator.apply(lambda x: x["single_global_effect_size"] / x["Curve Slope"], axis=1)
    df_curvecurator["Effect span > 1.5"] = df_curvecurator["Effect span"] > 1.5

    def is_in_range(value: float, target_range: np.ndarray):
        return target_range[0] <= value <= target_range[1]

    def get_double_fit_type(row: pd.Series):
        """
        Get the type of double fit.

        A pEC50 is called "substantial" if the step size is at least 10% of the global effect size.
        We can have one or two substantial pEC50s.

        We have the following types:
        - Target: All substantial pEC50s are in the target range.
        - Off-target: All substantial pEC50s are off the target range.
        - Both: One substantial pEC50 is in the target range, the other is off.
        - Weird: None of the pEC50s are substantial. This should never happen.
        """
        step1_in_range = is_in_range(row["double_pec50_1"], target_pec50_range)
        step2_in_range = is_in_range(row["double_pec50_2"], target_pec50_range)

        if step1_in_range and step2_in_range:
            return "Target"
        if not step1_in_range and not step2_in_range:
            return "Off-target"

        step1_substantial = row["double_pec50_1_substantial"]
        step2_substantial = row["double_pec50_2_substantial"]

        if step1_substantial and step2_substantial:
            return "Both"
        if step1_substantial:
            return "Target" if step1_in_range else "Off-target"
        if step2_substantial:
            return "Target" if step2_in_range else "Off-target"
        return "Weird"

    df_curvecurator["Double fit type"] = df_curvecurator.apply(lambda x: get_double_fit_type(x), axis=1)
    df_curvecurator["Single fit type"] = df_curvecurator["pEC50"].map(lambda x: "Target" if is_in_range(x, target_pec50_range) else "Off-target")

    return df_curvecurator

def plot_fit(df_curvecurator: pd.DataFrame, df_doses: pd.DataFrame, cell_line: str, target_ec50_range: np.ndarray = None, title: str = None):
    """
    Plot the fit of a single cell line.
    """
    cc_cell_line = df_curvecurator.T[cell_line]
    X = np.linspace(df_doses["pDose"].min(), df_doses["pDose"].max(), 1000)
    Y_single = __single_logistic__(X, *cc_cell_line["single_params"])
    Y_double = __double_logistic__(X, *cc_cell_line["double_params"])

    df_intensities = df_doses.join(df_curvecurator.T[cell_line])
    df_intensities.columns = df_doses.columns.tolist() + ["Intensity"]

    X = from_p_space(X)

    sns.scatterplot(data=df_intensities, x="Dose", y="Intensity", color="black")
    plt.plot(X, Y_single, color="gray", linestyle="--", label="Single-step fit")
    plt.plot(X, Y_double, color="red", label="Two-step fit")

    if target_ec50_range is not None:
        target_ec50 = from_p_space(np.mean(to_p_space(target_ec50_range)))
        plt.axvline(target_ec50, color="black", linestyle="-.", label="Target EC50")
        plt.axvline(target_ec50_range[0], color="gray", linestyle="-.", label="Target EC50 range")
        plt.axvline(target_ec50_range[1], color="gray", linestyle="-.")

    plt.title(title if title is not None else cell_line, fontsize=15)
    plt.xscale("log")
    plt.xlabel("Dose (M)")
    plt.ylabel("Relative intensity")

    plt.legend()
    sns.despine()

    plt.show()
    plt.close()

def plot_fit_type(df_curvecurator: pd.DataFrame, category: str, drug: str, palette: dict = None):
    """
    Plot the fit type distribution with respect to a given category.
    """
    df_plot = df_curvecurator[[category, "Double fit type"]].groupby([category, "Double fit type"]).size().unstack().fillna(0)
    df_plot.sort_values("Both", ascending=False, inplace=True)
    # Reorder columns to match desired order
    df_plot = df_plot[["Target", "Both", "Off-target"]]
    df_plot.plot.bar(stacked=True, color=[palette[col] for col in df_plot.columns] if palette is not None else None)

    # Set size dynamically
    plt.gcf().set_size_inches(0.5 * len(df_plot) + 2, 5)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)

    sns.despine()

    plt.ylabel("Number of cell lines", fontsize=15)
    plt.xlabel("Cancer type", fontsize=15)
    plt.title(f"{drug} - Fit type by {category.replace('_', ' ')}", fontsize=20)

def plot_paired_fit_type(df_curvecurator: pd.DataFrame, category: str, drug: str = None, palette: dict = None):
    df_grouping = pd.DataFrame(index=df_curvecurator.index)
    df_grouping["Single"] = df_curvecurator["Single fit type"]
    df_grouping["Double"] = df_curvecurator["Double fit type"]
    df_grouping["category"] = df_curvecurator[category]

    df_grouping = df_grouping.melt(id_vars="category", value_vars=["Single", "Double"], var_name="Fit type", value_name="Result")

    df_grouping = (df_grouping.groupby(["category", "Fit type", "Result"])
        .size()
        .reset_index(name="Count"))

    df_grouping = df_grouping.pivot_table(
        index=["category", "Fit type"], 
        columns="Result", 
        values="Count", 
        fill_value=0
    ).reset_index()

    # Calculate total entries per category and sort
    df_grouping["Total"] = df_grouping[["Both", "Off-target", "Target"]].sum(axis=1)
    category_order = (
        df_grouping.groupby("category")["Both"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    df_grouping["category"] = pd.Categorical(df_grouping["category"], categories=category_order, ordered=True)
    df_grouping = df_grouping.sort_values("category")

    # Prepare data for plotting
    categories = df_grouping["category"].unique()
    fit_types = ["Double", "Single"]
    results = ["Target", "Both", "Off-target"]  # Changed order here

    # Grouped bar plot settings
    bar_width = 0.4
    group_gap = 0.1  # Space between Single and Double bars
    category_gap = 0.6  # Space between different categories
    x_positions = []

    # Calculate x positions
    current_x = 0
    for _ in categories:
        for _ in fit_types:
            x_positions.append(current_x)
            current_x += bar_width + group_gap
        current_x += category_gap - group_gap

    # Define consistent colors
    if palette:
        colors = palette
    else:
        colors = {
            "Target": "#8da0cb",
            "Both": "#66c2a5",
            "Off-target": "#fc8d62"
        }

    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot bars
    for i, fit_type in enumerate(fit_types):
        subset = df_grouping[df_grouping["Fit type"] == fit_type]
        subset = subset.set_index("category").reindex(categories).fillna(0)
        bottom = np.zeros(len(x_positions) // len(fit_types))

        for result in results:
            ax.bar(
                x_positions[i::len(fit_types)],
                subset[result],
                bar_width,
                color=colors[result],
                label=result if i == 0 else None,
                bottom=bottom
            )
            bottom += subset[result]
    
    # Customization
    ax.set_xticks([np.mean(x_positions[i:i+len(fit_types)]) for i in range(0, len(x_positions), len(fit_types))])
    ax.set_xticklabels(categories, rotation=90)
    ax.set_ylabel("Count")
    category_pretty = category.capitalize().replace("_", " ")
    ax.set_xlabel(category_pretty)
    if drug:
        ax.set_title(f"{drug}: Fit Type Distribution across {category_pretty}s")
    else:
        ax.set_title(f"Fit Type Distribution across {category_pretty}s")
    ax.legend(title="EC50 Fit Type")

    # Adjust figure width based on number of categories
    fig.set_size_inches(0.2 * len(categories) + 2, 8)

    # Show plot
    plt.tight_layout()
    plt.show()
    plt.close()