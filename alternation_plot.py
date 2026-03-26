# -*- coding: utf-8 -*-
"""
Analysis of success rates in Y-/T-maze experiments.

This script:
- reads Excel data
- computes daily significance thresholds
- plots individual and group-level success rates (sex-means or cohort-mean)
- runs regression analyses
- compares selected experiment types
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import (
    binom,
    levene,
    mannwhitneyu,
    norm,
    pearsonr,
    probplot,
    shapiro,
    spearmanr,
    ttest_ind,
)


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.titlesize": 16,
    }
)


def style_axes(ax, ylabel="Success rate", title=None):
    """Apply consistent formatting to plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.set_ylabel(ylabel, fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=16)


def compute_daily_significance_threshold(df, experiment_type, alpha=0.05):
    """
    Compute significance thresholds for each age based on the average number
    of trials per rat.

    For spontaneous experiments, a two-tailed binomial threshold is used.
    For all other experiment types, a one-tailed threshold is used.
    """
    df_filtered = df[df["type of experiment"].str.lower() == experiment_type.lower()]
    threshold_by_day = {}

    for age, day_df in df_filtered.groupby("age"):
        n_trials_total = len(day_df)
        n_rats = day_df["ID"].nunique()
        n_trials_avg = n_trials_total / n_rats
        n_trials_rounded = int(round(n_trials_avg))

        if experiment_type.lower() == "spontaneous":
            min_successes = binom.isf(alpha / 2, n_trials_rounded, 0.5)
            max_failures = binom.ppf(alpha / 2, n_trials_rounded, 0.5)
            threshold_by_day[age] = {
                "lower_threshold": max_failures / n_trials_rounded,
                "upper_threshold": min_successes / n_trials_rounded,
            }
        else:
            min_successes = binom.isf(alpha, n_trials_rounded, 0.5)
            threshold_by_day[age] = min_successes / n_trials_rounded

    return threshold_by_day


def add_threshold_lines(ax, experiment_type, ages_sorted, thresholds_sorted):
    """Add significance threshold lines to a plot."""
    if experiment_type.lower() == "spontaneous":
        ax.plot(
            ages_sorted,
            thresholds_sorted["lower_threshold"],
            "g--",
            label="0.05 significance threshold",
        )
        ax.plot(ages_sorted, thresholds_sorted["upper_threshold"], "g--")
    else:
        ax.plot(
            ages_sorted,
            thresholds_sorted,
            "g--",
            label="0.05 significance threshold",
        )


def plot_individual_alternation(df, experiment_type, ages_sorted, thresholds_sorted):
    """Plot individual success rates for each animal."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    red_cmap = LinearSegmentedColormap.from_list(
        "trim_red", plt.get_cmap("Reds")(np.linspace(0.3, 0.9, 256))
    )
    blue_cmap = LinearSegmentedColormap.from_list(
        "trim_blue", plt.get_cmap("Blues")(np.linspace(0.3, 0.9, 256))
    )

    females = df[df["sex"] == "F"]["ID"].unique()
    males = df[df["sex"] == "M"]["ID"].unique()
    female_colors = red_cmap(np.linspace(0.2, 1, len(females))) if len(females) else []
    male_colors = blue_cmap(np.linspace(0.2, 1, len(males))) if len(males) else []

    for i, rat in enumerate(females):
        rat_data = df[df["ID"] == rat]
        ax.plot(rat_data["age"], rat_data["success_rate"], color=female_colors[i])

    for i, rat in enumerate(males):
        rat_data = df[df["ID"] == rat]
        ax.plot(rat_data["age"], rat_data["success_rate"], color=male_colors[i])

    add_threshold_lines(ax, experiment_type, ages_sorted, thresholds_sorted)
    ax.hlines(0.5, min(ages_sorted), max(ages_sorted), colors="grey", linestyles="--", label="Chance")

    ax.set_xlabel("Postnatal day", fontsize=16)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(linestyle="--", alpha=0.7)
    style_axes(ax, ylabel="Success rate", title="Individual success rate")
    plt.show()


def plot_sex_mean_alternation(df, experiment_type, ages_sorted, thresholds_sorted):
    """
    Plot mean success rate by sex with SEM.

    A Shapiro-Wilk test is used to check normality for each age.
    If both groups are normal, Welch's t-test is used.
    Otherwise, Mann-Whitney U is used.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    grouped = (
        df.groupby(["sex", "age"])
        .agg(
            mean_success_rate=("success_rate", "mean"),
            sem_success_rate=("success_rate", "sem"),
        )
        .reset_index()
    )

    colors = {"F": "firebrick", "M": "royalblue"}
    means_by_sex = {}
    all_ages = sorted(df["age"].unique())

    for sex in grouped["sex"].unique():
        sex_data = grouped[grouped["sex"] == sex]
        ages = sex_data["age"]
        mean = sex_data["mean_success_rate"]
        sem = sex_data["sem_success_rate"]

        means_by_sex[sex] = dict(zip(ages, mean))
        label = "Females" if sex == "F" else "Males"
        color = colors.get(sex, "gray")

        ax.plot(ages, mean, "-o", label=f"{label} (mean)", color=color)
        ax.fill_between(ages, mean - sem, mean + sem, color=color, alpha=0.2)

    for age in all_ages:
        vals_f = df[(df["sex"] == "F") & (df["age"] == age)]["success_rate"].dropna()
        vals_m = df[(df["sex"] == "M") & (df["age"] == age)]["success_rate"].dropna()

        if len(vals_f) > 2 and len(vals_m) > 2:
            p_f = shapiro(vals_f)[1]
            p_m = shapiro(vals_m)[1]

            if p_f > 0.05 and p_m > 0.05:
                _, p = ttest_ind(vals_f, vals_m, equal_var=False, nan_policy="omit")
                test_name = "t-test"
            else:
                _, p = mannwhitneyu(vals_f, vals_m, alternative="two-sided")
                test_name = "MWU"

            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = ""

            if stars:
                y_max = np.nanmax(
                    [
                        means_by_sex.get("F", {}).get(age, np.nan),
                        means_by_sex.get("M", {}).get(age, np.nan),
                    ]
                )
                ax.text(age, y_max + 0.05, stars, ha="center", va="bottom", fontsize=16)

            print(
                f"Age {age} | {test_name} used "
                f"(Shapiro pF={p_f:.3f}, pM={p_m:.3f}) -> p={p:.4f}"
            )
        else:
            print(f"Age {age} | Not enough data for normality test")

    xmin = df["age"].min()
    xmax = df["age"].max()
    add_threshold_lines(ax, experiment_type, ages_sorted, thresholds_sorted)
    ax.hlines(0.5, xmin, xmax, colors="grey", linestyles="--", label="Chance")

    ax.set_xlabel("Postnatal day", fontsize=16)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(linestyle="--", alpha=0.7)
    style_axes(ax, ylabel="Success rate", title="Mean success rate by sex")
    plt.show()


def plot_all_mean_alternation(df, experiment_type, ages_sorted, thresholds_sorted):
    """Plot mean and median success rate across age."""
    grouped = df.groupby("age")["success_rate"]
    ages = grouped.mean().index
    mean = grouped.mean().values
    median = grouped.median().values
    sem = grouped.sem().values

    print("\nMean success rate with SEM per age")
    for age, mean_value, sem_value in zip(ages, mean, sem):
        print(f"Age {age}: Mean = {mean_value:.3f}, SEM = {sem_value:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(ages, mean, "-o", color="black", label="Mean")
    ax.fill_between(ages, mean - sem, mean + sem, alpha=0.2, color="black")
    add_threshold_lines(ax, experiment_type, ages_sorted, thresholds_sorted)
    ax.hlines(0.5, xmin=ages[0], xmax=ages[-1], colors="grey", linestyles="--", label="Chance")

    ax.set_xlabel("Postnatal day", fontsize=16)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(linestyle="--", alpha=0.7)
    style_axes(ax, ylabel="Success rate", title="Mean success rate")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(ages, median, "-o", color="indigo", label="Median")
    add_threshold_lines(ax, experiment_type, ages_sorted, thresholds_sorted)
    ax.hlines(0.5, xmin=ages[0], xmax=ages[-1], colors="grey", linestyles="--", label="Chance")

    ax.set_xlabel("Postnatal day", fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(linestyle="--", alpha=0.7)
    style_axes(ax, ylabel="Success rate", title="Median success rate")
    plt.show()


def plot_regression_with_individuals_alternation(df, y_col, ylabel):
    """Plot individual data with linear regression and correlation statistics."""
    df_clean = df.dropna(subset=["age", y_col]).copy()
    X = sm.add_constant(df_clean["age"])
    y = df_clean[y_col]
    model = sm.OLS(y, X).fit()

    pearson_r, _ = pearsonr(df_clean["age"], df_clean[y_col])
    rho, p_spear = spearmanr(df_clean["age"], df_clean[y_col])

    age_range = np.linspace(df_clean["age"].min(), df_clean["age"].max(), 100)
    X_pred = sm.add_constant(age_range)
    preds = model.get_prediction(X_pred).summary_frame(alpha=0.05)

    r_squared = model.rsquared
    p_value = model.pvalues[1]
    slope = model.params[1]

    if p_value < 0.001:
        p_text = "< 0.001"
    elif p_value < 0.01:
        p_text = "< 0.01"
    elif p_value < 0.05:
        p_text = "< 0.05"
    else:
        p_text = f"= {p_value:.3f}"

    if p_spear < 0.001:
        p_text_spear = "< 0.001"
    elif p_spear < 0.01:
        p_text_spear = "< 0.01"
    elif p_spear < 0.05:
        p_text_spear = "< 0.05"
    else:
        p_text_spear = f"= {p_spear:.3f}"

    reg_text = (
        f"Slope = {slope:.3f}\n"
        f"R^2 = {r_squared:.2f}\n"
        f"Pearson r = {pearson_r:.2f}; p {p_text}\n"
        f"Spearman rho = {rho:.2f}; p {p_text_spear}"
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(df_clean["age"], df_clean[y_col], alpha=0.5, label="Individual data", color="black", s=30)
    ax.plot(age_range, preds["mean"], color="teal", label="Regression line")
    ax.fill_between(
        age_range,
        preds["mean_ci_lower"],
        preds["mean_ci_upper"],
        color="teal",
        alpha=0.2,
        label="95% CI",
    )

    ax.set_xlabel("Postnatal day", fontsize=16)
    ax.set_ylim(-0.02, 1.02)
    ax.text(
        0.02,
        0.03,
        reg_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round,pad=0.4"),
    )
    ax.grid(alpha=0.6, linestyle="--")
    ax.legend(fontsize=14, loc="lower right")
    style_axes(ax, ylabel=ylabel, title="Regression between success rate and age")
    plt.tight_layout()
    plt.show()


def check_normality_visual(data, exp, day):
    """Run Shapiro-Wilk and show a Q-Q plot for visual inspection."""
    if len(data) < 3:
        print(f"Not enough data for visual inspection ({exp}, day {day})")
        return False

    _, p = shapiro(data)
    normal = p > 0.05
    status = "Normal" if normal else "Non-normal"
    print(f"{exp} - Day {day}: Shapiro p={p:.4f} -> {status}")

    plt.figure(figsize=(4, 4), dpi=200)
    probplot(data, dist="norm", plot=plt)
    plt.title(f"{exp} - Day {day}\nShapiro p={p:.4f} ({status})", fontsize=10)
    plt.tight_layout()
    plt.show()

    return normal


def plot_compare_experiments_graph(df, exp1, exp2, selected_days, use_day_index=True):
    """
    Compare mean success rates for two experiment types across selected days.

    For each day, normality is checked before selecting either a t-test
    or a Mann-Whitney U test.
    """
    df = df.copy()
    df["experiment_type"] = df["experiment_type"].str.lower()
    exp1 = exp1.lower()
    exp2 = exp2.lower()

    x_col = "day_index" if use_day_index else "age"
    grouped = df.groupby(["experiment_type", x_col])["success_rate"]

    means_exp1, sems_exp1, means_exp2, sems_exp2, p_values = [], [], [], [], []

    for day in selected_days:
        vals1 = grouped.get_group((exp1, day)).dropna() if (exp1, day) in grouped.groups else []
        vals2 = grouped.get_group((exp2, day)).dropna() if (exp2, day) in grouped.groups else []

        means_exp1.append(np.mean(vals1) if len(vals1) > 0 else np.nan)
        sems_exp1.append(np.std(vals1, ddof=1) / np.sqrt(len(vals1)) if len(vals1) > 1 else np.nan)
        means_exp2.append(np.mean(vals2) if len(vals2) > 0 else np.nan)
        sems_exp2.append(np.std(vals2, ddof=1) / np.sqrt(len(vals2)) if len(vals2) > 1 else np.nan)

        if len(vals1) > 2 and len(vals2) > 2:
            normal1 = check_normality_visual(vals1, exp1, day)
            normal2 = check_normality_visual(vals2, exp2, day)

            if normal1 and normal2:
                _, p = ttest_ind(vals1, vals2, equal_var=False, nan_policy="omit")
            else:
                _, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
        else:
            p = np.nan

        p_values.append(p)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    means_exp1 = np.array(means_exp1)
    sems_exp1 = np.array(sems_exp1)
    means_exp2 = np.array(means_exp2)
    sems_exp2 = np.array(sems_exp2)

    ax.plot(selected_days, means_exp1, "-o", color="#0041C2", label=exp1)
    ax.fill_between(selected_days, means_exp1 - sems_exp1, means_exp1 + sems_exp1, color="#0041C2", alpha=0.2)

    ax.plot(selected_days, means_exp2, "-o", color="#006A4E", label=exp2)
    ax.fill_between(selected_days, means_exp2 - sems_exp2, means_exp2 + sems_exp2, color="#006A4E", alpha=0.2)

    for x, y1, y2, p in zip(selected_days, means_exp1, means_exp2, p_values):
        if not np.isnan(p):
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = ""

            if stars:
                y_max = np.nanmax([y1, y2])
                ax.text(x, y_max + 0.05, stars, ha="center", va="bottom", fontsize=16, color="black")

    ax.hlines(0.5, xmin=selected_days[0], xmax=selected_days[-1], colors="grey", linestyles="--", label="Chance")
    ax.set_xlabel("Training day" if use_day_index else "Postnatal day", fontsize=16)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right", fontsize=14)
    ax.grid(alpha=0.6, linestyle="--")
    style_axes(ax, ylabel="Success rate", title="Comparison of success rates")
    plt.tight_layout()
    plt.show()





# File paths
data_path = Path(r"C:\Users\solho\OneDrive - Universitetet i Oslo\Documents\Essentials\Data\alternation_with_time_all.xlsx")
ymaze_file = Path(r"C:\Users\solho\Desktop\info y maze wo empty.xlsx")
output_dir = Path(r"C:\Users\solho\Desktop\y maze\alternation")


# Load data
# df contains summary-level data
# df_all contains trial-level data used for threshold calculations
# output_dir is kept here in case you want to save figures later

df = pd.read_excel(data_path)
df_all = pd.read_excel(ymaze_file)


# Select experiment types to analyze
experiment_types = ["sd"]

for experiment_type in experiment_types:
    filtered_df = df[df["experiment_type"].str.lower() == experiment_type.lower()]
    threshold_by_day = compute_daily_significance_threshold(df_all, experiment_type, alpha=0.05)
    ages_sorted = sorted(threshold_by_day.keys())

    if experiment_type.lower() == "spontaneous":
        thresholds_sorted = {
            "lower_threshold": [threshold_by_day[age]["lower_threshold"] for age in ages_sorted],
            "upper_threshold": [threshold_by_day[age]["upper_threshold"] for age in ages_sorted],
        }
    else:
        thresholds_sorted = [threshold_by_day[age] for age in ages_sorted]

    plot_individual_alternation(filtered_df, experiment_type, ages_sorted, thresholds_sorted)
    plot_sex_mean_alternation(filtered_df, experiment_type, ages_sorted, thresholds_sorted)
    plot_all_mean_alternation(filtered_df, experiment_type, ages_sorted, thresholds_sorted)
    plot_regression_with_individuals_alternation(
        df=filtered_df,
        y_col="success_rate",
        ylabel="Success rate",
    )


# Compare experiments
# day_index is the relative training day for each animal within each experiment type

df = df.sort_values(["experiment_type", "ID", "age"])
df["day_index"] = df.groupby(["experiment_type", "ID"]).cumcount() + 1

exp1 = "sd"
exp2 = "regular_DNMP"
selected_days = [23, 24, 25, 26, 27, 28, 29, 30]

plot_compare_experiments_graph(df, exp1, exp2, selected_days, use_day_index=False)
