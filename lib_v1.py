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
    df_curvecurator["EC50"] = 1e3 * (10 ** -df_curvecurator["pEC50"])

    # Remove rows from df_doses, where the corresponding column in curvecurator has only one value
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

def plot_fit(df_curvecurator: pd.DataFrame, df_doses: pd.DataFrame, cell_line: str, title: str = None):
    pred_fun = __double_logistic__

    x = df_doses["Dose"]
    y = df_curvecurator[df_doses.index].loc[cell_line]

    r2 = df_curvecurator.loc[cell_line]["double_r2"]
    params = df_curvecurator.loc[cell_line]["double_params"]

    front, mid_ratio, back_ratio, slope1, slope2, pec50_1, pec50_delta = params
    front, middle, back, slope1, slope2, pec50_1, pec50_2 = __get_actual_parameters__(front, mid_ratio, back_ratio, slope1, slope2, pec50_1, pec50_delta)

    x_center = 10 ** ((np.log10(pec50_1) + np.log10(pec50_2)) / 2)

    y_max = pred_fun(0, *params)
    y_min = pred_fun(np.inf, *params)
    y_center = pred_fun(x_center, *params)

    target_effect = y_max - y_center

    plt.scatter(x, y, label='Data', color='blue')
    x_gen = np.logspace(np.log10(X_MIN), np.log10(x.max()), 10000)
    plt.plot(x_gen, pred_fun(x_gen, *params), label='Fitted Model', color='red')

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

def __fit_double_logistic__(x:pd.Series, y:pd.Series, target_ec50_μm:float):
    mask = y.notna()
    x = x[mask].values
    y = y[mask].values

    target_ec50_min = max(target_ec50_μm / 10, X_MIN)
    target_ec50_max = target_ec50_μm * 10

    popt_pos, _ = curve_fit(__double_logistic__, x, y,
                        maxfev=int(1e6),
                        p0=[1, 0.5, 0.5, 1, 1e-3, (target_ec50_max + target_ec50_min) / 2, 2],
                        method='trf',
                        bounds=(
                            [0.7, 0.01,   0,  1, 1e-5, target_ec50_min, 1],
                            [1.5 , 0.8 , 0.8, 4,    4, target_ec50_max, 5]
                        ))
    popt_pos_r2 = r2_score(y, __double_logistic__(x, *popt_pos))

    popt_neg, _ = curve_fit(__double_logistic__, x, y,
                        maxfev=int(1e6),
                        p0=[1, 0.5, 0.5, 1, 1e-3, (target_ec50_max + target_ec50_min) / 2, -2],
                        method='trf',
                        bounds=(
                            [0.7, 0.01,   0,  1, 1e-5, target_ec50_min, -5],
                            [1.5 , 0.8 , 0.8, 4,    4, target_ec50_max, -1]
                        ))
    popt_neg_r2 = r2_score(y, __double_logistic__(x, *popt_neg))

    if popt_pos_r2 > popt_neg_r2:
        return popt_pos_r2, popt_pos
    else:
        return popt_neg_r2, popt_neg

def fit(df_curvecurator: pd.DataFrame, df_doses: pd.DataFrame, target_ec50_μm:float):
    df_curvecurator["single_r2"] = df_curvecurator["Curve R2"]
    df_curvecurator["double_fit_res"] = df_curvecurator[df_doses.index].apply(lambda x: __fit_double_logistic__(df_doses["Dose"], x, target_ec50_μm), axis=1)
    df_curvecurator["double_r2"] = df_curvecurator["double_fit_res"].apply(lambda x: x[0])
    df_curvecurator["double_params"] = df_curvecurator["double_fit_res"].apply(lambda x: x[1])
    df_curvecurator["double_pec50_1"] = df_curvecurator["double_params"].apply(lambda x: x[5])
    df_curvecurator["double_pec50_2"] = df_curvecurator["double_params"].apply(lambda x: x[5] + 10 ** x[6])
    df_curvecurator["double_x_center"] = df_curvecurator.apply(lambda x: 10 ** ((np.log10(x["double_pec50_1"]) + np.log10(x["double_pec50_2"])) / 2), axis=1)
    df_curvecurator["double_y_max"] = df_curvecurator.apply(lambda x: __double_logistic__(0, *x["double_params"]), axis=1)
    df_curvecurator["double_y_center"] = df_curvecurator.apply(lambda x: __double_logistic__(x["double_x_center"], *x["double_params"]), axis=1)

    df_curvecurator["Global effect"] = 2 ** df_curvecurator["Curve Fold Change"]
    df_curvecurator["sigmoid_diff"] = df_curvecurator["double_r2"] - df_curvecurator["single_r2"]
    df_curvecurator["two_step_better"] = df_curvecurator["sigmoid_diff"] > 0
    df_curvecurator["single_step_is_target"] = (target_ec50_µm < (1e3 * df_curvecurator["EC50"] * 5)) & (target_ec50_µm > (1e3 * df_curvecurator["EC50"] / 5))

    df_curvecurator["Fit type"] = df_curvecurator.apply(lambda x: "Both" if x["two_step_better"] else "Target" if x["single_step_is_target"] else "Off-target", axis=1)

    def get_target_effect_size(row):
        if row["Fit type"] == "Both":
            return row["double_y_max"] - row["double_y_center"]
        elif row["Fit type"] == "Target":
            return row["Global effect"]
        else:
            return 0

    df_curvecurator["target_effect_size"] = df_curvecurator.apply(get_target_effect_size, axis=1)


    return df_curvecurator

def plot_fit_type(df_curvecurator: pd.DataFrame, category: str, drug: str = None, palette: dict = None):
    df_grouping = pd.DataFrame(index=df_curvecurator.index)
    df_grouping["Single"] = df_curvecurator["single_step_is_target"].map({True: "Target", False: "Off-target"})
    df_grouping["Double"] = df_curvecurator["Fit type"].map(lambda x: x.capitalize().replace("_", "-"))
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
    results = ["Both", "Off-target", "Target"]

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
            "Both": "#66c2a5",
            "Off-target": "#fc8d62",
            "Target": "#8da0cb"
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
