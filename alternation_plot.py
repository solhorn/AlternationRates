# -*- coding: utf-8 -*-
"""
Created on Mon Jun 9 11:28:54 2025
@author: solvehor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.stats import binom
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.stats import levene, bartlett
from scipy.stats import mannwhitneyu



plt.rcParams.update({
    "font.size": 12,        # standard for all text
    "axes.labelsize": 14,   # axis labels
    "xtick.labelsize": 14,  # numbers on x axis
    "ytick.labelsize": 14,  # numbers on y axis
    "legend.fontsize": 12,  # legend
    "axes.titlesize": 16    # title
})


### PREPARE DATA ##

def compute_daily_significance_threshold(df, experiment_type, alpha=0.05):
    """
    Computes significance tresholds for each day, based on average number 
    of trials per day.
    One-tailed binomial test for all experiments, except from the spontaneous
    alternation control experiment. 
    """
    df_filtered = df[df["type of experiment"].str.lower() == experiment_type.lower()]
    threshold_by_day = {}

    for age, day_df in df_filtered.groupby("age"):
        n_trials_total = len(day_df)
        n_rats = day_df["ID"].nunique()

        n_trials_avg = n_trials_total / n_rats
        n_trials_rounded = int(round(n_trials_avg))

        # two-tailed binomial test for the spontaneous control experiment:
        if experiment_type == 'spontaneous':
            # Right tail - minimum successes for significans
            min_successes = binom.isf(alpha / 2, n_trials_rounded, 0.5)
            
            # left tail - maximum successes for sicnificans
            max_failures = binom.ppf(alpha / 2, n_trials_rounded, 0.5)
            
            threshold_by_day[age] = {
                "lower_threshold": max_failures / n_trials_rounded,
                "upper_threshold": min_successes / n_trials_rounded
            }

        # one-tailed binomial test for all other experiment types:
        else:
            min_successes = binom.isf(alpha, n_trials_rounded, 0.5)
            threshold = min_successes / n_trials_rounded
            threshold_by_day[age] = threshold

    return threshold_by_day




### PLOTS ###


def plot_individual_alternation(df, directory, experiment_type, ages_sorted, thresholds_sorted):
    """
    Plots individual alternation rates for each rat in the same plot.
    Female rats get colors from a red colormap.
    Male rats get colors from a blue colormap.
    """
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # colormaps
    red_cmap = LinearSegmentedColormap.from_list("trim_red", plt.get_cmap("Reds")(np.linspace(0.3, 0.9, 256)))
    blue_cmap = LinearSegmentedColormap.from_list("trim_blue", plt.get_cmap("Blues")(np.linspace(0.3, 0.9, 256)))

    # Get rats per sex
    females = df[df["sex"] == "F"]["ID"].unique()
    males = df[df["sex"] == "M"]["ID"].unique()
    cmap_f = red_cmap(np.linspace(0.2, 1, len(females)))
    cmap_m = blue_cmap(np.linspace(0.2, 1, len(males)))

    # Plot females with one colormap
    for i, rat in enumerate(females):
        rat_data = df[df['ID'] == rat]
        ax.plot(rat_data['age'], rat_data['success_rate'], color=cmap_f[i])

    # Plot males with the other colormapp
    for i, rat in enumerate(males):
        rat_data = df[df['ID'] == rat]
        ax.plot(rat_data['age'], rat_data['success_rate'], color=cmap_m[i])

    # Add significance thresholds
    if experiment_type == 'spontaneous':
        ax.plot(ages_sorted, thresholds_sorted['lower_threshold'], 'g--', label='0.05 significance threshold')
        ax.plot(ages_sorted, thresholds_sorted['upper_threshold'], 'g--')
    else:
        ax.plot(ages_sorted, thresholds_sorted, 'g--', label='0.05 significance threshold')

    # Add Chance line in plot    
    ax.hlines(0.5, min(ages_sorted), max(ages_sorted), colors="grey", linestyles="--", label="Chance")
    
    ax.legend(fontsize=14, loc="lower right")
    ax.set_xlabel('Postnatal day', fontsize=16)
    ax.set_ylabel('Alternation rate', fontsize=16)
    ax.set_ylim(0,1)
    ax.set_title(f'Individual alternation rate', fontsize=16)
    # ax.set_ylim(0.31, 1.01)
    

    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Individual success rate', fontsize=16)



    ax.grid(linestyle='--', alpha=0.7)


    # file_path = os.path.join(directory, f"individual_alternation_{experiment_type}.png")
    # plt.savefig(file_path)
    plt.show()



def plot_sex_mean_alternation(df, directory, experiment_type, ages_sorted, thresholds_sorted):
    """
    Plots mean DNMP success rate by sex with SEM,
    and indicates significant differences between sexes for each day based on 
    statistical test with Mann Whitney U-test.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    
    # for each combination of sex and age, mean success rate and SEM is calculated:
    grouped = df.groupby(['sex', 'age']).agg(
        mean_success_rate=('success_rate', 'mean'),
        sem_success_rate=('success_rate', 'sem')
    ).reset_index()


    colors = {'F': 'firebrick', 'M': 'royalblue'}
    means_by_sex = {}
    sems_by_sex = {}
    all_ages = sorted(df['age'].unique())

    # Plot mean +- SEM
    for sex in grouped['sex'].unique():
        sex_data = grouped[grouped['sex'] == sex]
        ages = sex_data['age']
        mean = sex_data['mean_success_rate']
        sem = sex_data['sem_success_rate']

        means_by_sex[sex] = dict(zip(ages, mean))
        sems_by_sex[sex] = dict(zip(ages, sem))

        symbol = "Females" if sex == "F" else "Males"
        plt.plot(ages, mean, "-o", label=f"{symbol} (mean)", color=colors.get(sex, 'gray'))
        plt.fill_between(ages, mean - sem, mean + sem, color=colors.get(sex, 'gray'), alpha=0.2)


    # statistical test between sexes for each day:
    p_values = []
    for age in all_ages:
        vals_f = df[(df['sex'] == 'F') & (df['age'] == age)]['success_rate']
        vals_m = df[(df['sex'] == 'M') & (df['age'] == age)]['success_rate']

        if len(vals_f) > 1 and len(vals_m) > 1:
            
            # Using t-test:
            # t, p = ttest_ind(vals_f, vals_m, equal_var=False, nan_policy='omit')
            
            # Using Mann Whitney U test:
            u, p = mannwhitneyu(vals_f, vals_m, alternative="two-sided")
            
            p_values.append((age, p))
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = ''
            
            y_max = np.nanmax([
                means_by_sex['F'].get(age, np.nan),
                means_by_sex['M'].get(age, np.nan)
            ])
            plt.text(age, y_max + 0.05, stars, ha='center', va='bottom',
                     fontsize=16, color='black')

        else:
            p_values.append((age, np.nan))

    # print(p_values)

    xmin = df['age'].min()
    xmax = df['age'].max()

    # Significance thresholds
    if experiment_type == 'spontaneous':
        plt.plot(ages_sorted, thresholds_sorted['lower_threshold'], 'g--', label='0.05 significance threshold')
        plt.plot(ages_sorted, thresholds_sorted['upper_threshold'], 'g--')
    else:
        plt.plot(ages_sorted, thresholds_sorted, 'g--', label='0.05 significance threshold')

    # Add Chance line
    plt.hlines(0.5, xmin, xmax, colors="grey", linestyles="--", label="Chance")

    plt.xlabel('Postnatal day', fontsize=16)
    plt.ylabel('Alternation rate', fontsize=16)
    plt.title(f'Mean alternation rate by sex', fontsize=16)
    plt.legend(fontsize=14, loc="lower right")
    plt.grid(linestyle='--', alpha=0.7)
    plt.ylim(0, 1.1)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Mean success rate by sex', fontsize=16)
    
    
    # file_path = os.path.join(directory, f"sex_mean_alternation_{experiment_type}.png")
    # plt.savefig(file_path)
    plt.show()


from scipy.stats import ttest_ind, mannwhitneyu, shapiro

def plot_sex_mean_alternation(df, directory, experiment_type, ages_sorted, thresholds_sorted):
    """
    Plots mean DNMP success rate by sex with SEM.
    Performs statistical comparison between sexes (per age),
    automatically choosing between independent samples t-test and Mann–Whitney U-test
    based on Shapiro–Wilk normality test for each cohort.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Mean and SEM for each sex × age
    grouped = df.groupby(['sex', 'age']).agg(
        mean_success_rate=('success_rate', 'mean'),
        sem_success_rate=('success_rate', 'sem')
    ).reset_index()

    colors = {'F': 'firebrick', 'M': 'royalblue'}
    means_by_sex = {}
    sems_by_sex = {}
    all_ages = sorted(df['age'].unique())

    # Plot mean ± SEM
    for sex in grouped['sex'].unique():
        sex_data = grouped[grouped['sex'] == sex]
        ages = sex_data['age']
        mean = sex_data['mean_success_rate']
        sem = sex_data['sem_success_rate']

        means_by_sex[sex] = dict(zip(ages, mean))
        sems_by_sex[sex] = dict(zip(ages, sem))

        symbol = "Females" if sex == "F" else "Males"
        plt.plot(ages, mean, "-o", label=f"{symbol} (mean)", color=colors.get(sex, 'gray'))
        plt.fill_between(ages, mean - sem, mean + sem, color=colors.get(sex, 'gray'), alpha=0.2)

    # Statistical test per age (automatic normality check)
    p_values = []
    for age in all_ages:
        vals_f = df[(df['sex'] == 'F') & (df['age'] == age)]['success_rate'].dropna()
        vals_m = df[(df['sex'] == 'M') & (df['age'] == age)]['success_rate'].dropna()

        if len(vals_f) > 2 and len(vals_m) > 2:  # trenger minst 3 per gruppe for Shapiro
            # --- 🔹 Normality check ---
            p_f = shapiro(vals_f)[1]
            p_m = shapiro(vals_m)[1]
            normal_f = p_f > 0.05
            normal_m = p_m > 0.05
            both_normal = normal_f and normal_m

            # --- 🔹 Choose appropriate test ---
            if both_normal:
                test_name = "t-test"
                stat, p = ttest_ind(vals_f, vals_m, equal_var=False, nan_policy='omit')
            else:
                test_name = "MWU"
                stat, p = mannwhitneyu(vals_f, vals_m, alternative="two-sided")

            p_values.append((age, p))

            # --- 🔹 Significance stars ---
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                stars = ''

            # Plot stars above the higher mean value
            y_max = np.nanmax([
                means_by_sex['F'].get(age, np.nan),
                means_by_sex['M'].get(age, np.nan)
            ])
            if stars:
                plt.text(age, y_max + 0.05, stars, ha='center', va='bottom',
                         fontsize=16, color='black')

            # --- 🔹 Debug printout ---
            print(f"Age {age} | {test_name} used "
                  f"(Shapiro pF={p_f:.3f}, pM={p_m:.3f}) → p={p:.4f}")

        else:
            p_values.append((age, np.nan))
            print(f"Age {age} | Not enough data for normality test")

    # --- Plot decorations ---
    xmin = df['age'].min()
    xmax = df['age'].max()

    # Significance thresholds
    if experiment_type == 'spontaneous':
        plt.plot(ages_sorted, thresholds_sorted['lower_threshold'], 'g--', label='0.05 significance threshold')
        plt.plot(ages_sorted, thresholds_sorted['upper_threshold'], 'g--')
    else:
        plt.plot(ages_sorted, thresholds_sorted, 'g--', label='0.05 significance threshold')

    # Chance line
    plt.hlines(0.5, xmin, xmax, colors="grey", linestyles="--", label="Chance")

    plt.xlabel('Postnatal day', fontsize=16)
    plt.ylabel('Alternation rate', fontsize=16)
    plt.title(f'Mean alternation rate by sex', fontsize=16)
    plt.legend(fontsize=14, loc="lower right")
    plt.grid(linestyle='--', alpha=0.7)
    plt.ylim(0, 1.1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    
    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Mean success rate by sex', fontsize=16)

    # Save and show
    # file_path = os.path.join(directory, f"sex_mean_alternation_{experiment_type}.png")
    # plt.savefig(file_path)
    plt.show()



def plot_all_mean_alternation(df, directory, experiment_type, ages_sortes, thresholds_sorted):
    """
    Plots mean and median DNMP success rate across days.
    """
    # Find info
    grouped = df.groupby('age')['success_rate']
    ages = grouped.mean().index
    mean = grouped.mean().values
    median = grouped.median().values
    sem = grouped.sem().values

    # Print age, mean og SEM
    print("\n=== Mean alternation rate with SEM per age ===")
    for a, m, s in zip(ages, mean, sem):
        print(f"Age {a}: Mean = {m:.3f}, SEM = {s:.3f}")


    # PLOT MEAN
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(ages, mean, '-o', color='black', label='Mean')
    plt.fill_between(ages, mean - sem, mean + sem, alpha=0.2, color='black')
    
    # sd: 0041C2 
    
    
    # Significance thresholds
    if experiment_type == 'spontaneous':
        plt.plot(ages_sorted, thresholds_sorted['lower_threshold'], 'g--', label='0.05 significance threshold')
        plt.plot(ages_sorted, thresholds_sorted['upper_threshold'], 'g--')
    else:
        plt.plot(ages_sorted, thresholds_sorted, 'g--', label='0.05 significance threshold')
    
    # Add Chance line
    plt.hlines(0.5, xmin=ages[0], xmax=ages[-1], colors="grey", linestyles="--", label='Chance')

    plt.xlabel('Postnatal day', fontsize=16)
    plt.ylabel('Alternation rate', fontsize=16)
    plt.title(f'Mean alternation rate', fontsize=16)
    plt.legend(fontsize=14, loc="lower right")
    plt.grid(linestyle='--', alpha=0.7)
    # plt.ylim(0.31, 1.01)
    plt.ylim(0, 1)

    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Mean success rate', fontsize=16)
    
    
    # file_path = os.path.join(directory, f"mean_alternation_{experiment_type}.png")
    # plt.savefig(file_path)
    plt.show()



    # PLOT MEDIAN
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(ages, median, '-o', color='indigo', label='Median')
    
    if experiment_type == 'spontaneous':
        plt.plot(ages_sorted, thresholds_sorted['lower_threshold'], 'g--', label='0.05 significance threshold')
        plt.plot(ages_sorted, thresholds_sorted['upper_threshold'], 'g--')
    else:
        plt.plot(ages_sorted, thresholds_sorted, 'g--', label='0.05 significance threshold')
        
    plt.hlines(0.5, xmin=ages[0], xmax=ages[-1], colors="grey", linestyles="--", label='Chance')

    plt.xlabel('Postnatal day', fontsize=14)
    plt.ylabel('Alternation rate', fontsize=14)
    plt.title(f'Median success rate', fontsize=16)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(linestyle='--', alpha=0.7)
    # plt.ylim(0.31, 1.01)
    plt.ylim(0, 1)

    # file_path = os.path.join(directory, f"median_alternation_{experiment_type}.png")
    # plt.savefig(file_path)
    plt.show()



from scipy.stats import spearmanr

def plot_regression_with_individuals_alternation(df, y_col, ylabel, directory, experiment_type, filename_suffix):
    """
    Plots individual data points with regression line and 95% CI.
    """
    df_clean = df.dropna(subset=['age', y_col]).copy()
    X = sm.add_constant(df_clean['age'])
    y = df_clean[y_col]
    model = sm.OLS(y, X).fit() # Runs an Ordinary Least Squares (OLS) regression between X and y (from statsmodels)

    from scipy.stats import pearsonr
    pearson_r, pearson_p = pearsonr(df_clean['age'], df_clean[y_col])

    # Make predictions + SEM
    age_range = np.linspace(df_clean['age'].min(), df_clean['age'].max(), 100)
    X_pred = sm.add_constant(age_range)
    preds = model.get_prediction(X_pred).summary_frame(alpha=0.05)  # predictions and 95% CI

    # Get model info
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
    # reg_text = f"Slope = {slope:.3f}\n$R^2$ = {r_squared:.2f}\np {p_text}"

    # --- Spearman correlation ---
    rho, p_spear = spearmanr(df_clean['age'], df_clean[y_col])

    if p_spear < 0.001:
        p_text_spear = "< 0.001"
    elif p_spear < 0.01:
        p_text_spear = "< 0.01"
    elif p_spear < 0.05:
        p_text_spear = "< 0.05"
    else:
        p_text_spear = f"= {p_spear:.3f}"

    # --- Text box content ---
    reg_text = (
        f"Slope = {slope:.3f}\n"
        f"R² = {r_squared:.2f}\n"
        f"Pearson's r = {pearson_r:.2f}; p {p_text}\n"
        f"Spearman's ρ = {rho:.2f}; p {p_text_spear}"
    )

#     reg_text = (
#     f"Slope = {slope:.3f}\n"
#     f"$R^2$ = {r_squared:.2f}\n"
#     f"Pearson's r = {pearson_r:.2f}\n"
#     f"p {p_text}"
# )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(df_clean['age'], df_clean[y_col], alpha=0.5, label='Individual data', color='black', s=30)
    ax.plot(age_range, preds['mean'], color='teal', label='Regression line')
    ax.fill_between(age_range, preds['mean_ci_lower'], preds['mean_ci_upper'],
                    color='teal', alpha=0.2, label='95% CI')

    ax.set_xlabel('Postnatal day', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(f'Regression between alternation rate and age', fontsize=17)
    ax.text(0.02, 0.03, reg_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='bottom',
            bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.4'))
    ax.grid(alpha=0.6, linestyle='--')
    ax.legend(fontsize=14, loc='lower right')
    ax.set_ylim(-0.02, 1.02)

    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Regression between success rate and age', fontsize=16)
    
    # file_path = os.path.join(directory, f"{filename_suffix}_{experiment_type}.png")
    plt.tight_layout()
    # plt.savefig(file_path)
    plt.show()


from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_spearman_with_individuals_alternation(df, y_col, ylabel, directory, experiment_type, filename_suffix):
    """
    Plots individual data points with regression line and 95% CI.
    Shows:
    - slope
    - R² from linear regression
    - Spearman rho + p
    """
    df_clean = df.dropna(subset=['age', y_col]).copy()
    X = sm.add_constant(df_clean['age'])
    y = df_clean[y_col]
    model = sm.OLS(y, X).fit()

    # --- Predictions ---
    age_range = np.linspace(df_clean['age'].min(), df_clean['age'].max(), 100)
    X_pred = sm.add_constant(age_range)
    preds = model.get_prediction(X_pred).summary_frame(alpha=0.05)

    # --- Linear regression info ---
    slope = model.params[1]
    r_squared = model.rsquared
    p_value_lin = model.pvalues[1]

    # Pretty p-value formatting
    if p_value_lin < 0.001:
        p_text_lin = "< 0.001"
    elif p_value_lin < 0.01:
        p_text_lin = "< 0.01"
    elif p_value_lin < 0.05:
        p_text_lin = "< 0.05"
    else:
        p_text_lin = f"= {p_value_lin:.3f}"

    # --- Spearman correlation ---
    rho, p_spear = spearmanr(df_clean['age'], df_clean[y_col])

    if p_spear < 0.001:
        p_text_spear = "< 0.001"
    elif p_spear < 0.01:
        p_text_spear = "< 0.01"
    elif p_spear < 0.05:
        p_text_spear = "< 0.05"
    else:
        p_text_spear = f"= {p_spear:.3f}"

    # --- Text box content ---
    reg_text = (
        f"Slope = {slope:.3f}\n"
        f"R² = {r_squared:.2f}\n"
        f"Spearman's ρ = {rho:.2f}\n"
        f"p {p_text_spear}"
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(df_clean['age'], df_clean[y_col], alpha=0.5, color='black', s=30, label='Individual data')

    ax.plot(age_range, preds['mean'], color='teal', label='Regression line')
    ax.fill_between(age_range, preds['mean_ci_lower'], preds['mean_ci_upper'],
                    color='teal', alpha=0.2, label='95% CI')

    ax.set_xlabel('Postnatal day', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title('Regression between alternation rate and age', fontsize=17)

    ax.text(0.02, 0.03, reg_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='bottom',
            bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.4'))

    ax.grid(alpha=0.6, linestyle='--')
    ax.legend(fontsize=14, loc='lower right')
    ax.set_ylim(-0.02, 1.02)

    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Regression between success rate and age', fontsize=16)
    
    # file_path = os.path.join(directory, f"{filename_suffix}_{experiment_type}.png")
    plt.tight_layout()
    # plt.savefig(file_path)
    plt.show()


### COMPARE AND PLOT DATA FROM TWO SELECTED EXPERIMENTS

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, norm, probplot

def check_normality_visual(data, label, day, exp):
    """
    Performs Shapiro–Wilk test and visualizes the distribution with a histogram and Q–Q plot.
    """
    if len(data) < 3:
        print(f"Not enough data for visual inspection ({exp}, day {day})")
        return

    # Shapiro–Wilk test
    stat, p = shapiro(data)
    normal = p > 0.05

    print(f"\n{exp} – Day {day}: Shapiro p={p:.4f} → {'Normal' if normal else 'Non-normal'}")

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"{exp} – Day {day}\nShapiro p={p:.4f} ({'Normal' if normal else 'Non-normal'})")

    # Histogram with normal curve
    axs[0].hist(data, bins=8, color="#5C3317", alpha=0.7, edgecolor='black', density=True)
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    axs[0].plot(x, norm.pdf(x, mu, sigma), 'k--', lw=1.5, label='Normal curve')
    axs[0].set_title("Histogram")
    axs[0].legend()

    # Q–Q plot
    probplot(data, dist="norm", plot=axs[1])
    axs[1].get_lines()[1].set_color('red')  # regression line
    axs[1].set_title("Q–Q Plot")

    plt.tight_layout()
    plt.show()

    return normal


def plot_compare_experiments_graph(df, directory, exp1, exp2, selected_days,
                                   use_day_index=True):
    """
    Plots mean alternation rate for two experiment types.
    Can use either absolute postnatal age or relative day_index.
    Performs statistical test for each selected day and marks significance with stars.
    Shows SEM as shaded area.
    """
    df = df.copy()
    df["experiment_type"] = df["experiment_type"].str.lower()
    exp1, exp2 = exp1.lower(), exp2.lower()

    # choose axis
    x_col = "day_index" if use_day_index else "age"

    grouped = df.groupby(['experiment_type', x_col])['success_rate']

    means_exp1, sems_exp1 = [], []
    means_exp2, sems_exp2 = [], []
    p_values = []

    for day in selected_days:
        try:
            vals1 = grouped.get_group((exp1, day))
        except KeyError:
            vals1 = []
            print(f"No data for {exp1} on {x_col}={day}")
        try:
            vals2 = grouped.get_group((exp2, day))
        except KeyError:
            vals2 = []
            print(f"No data for {exp2} on {x_col}={day}")

        # Find Mean and SEM values for each experiment
        means_exp1.append(np.mean(vals1) if len(vals1) > 0 else np.nan)
        sems_exp1.append(np.std(vals1, ddof=1)/np.sqrt(len(vals1)) if len(vals1) > 1 else np.nan)
        means_exp2.append(np.mean(vals2) if len(vals2) > 0 else np.nan)
        sems_exp2.append(np.std(vals2, ddof=1)/np.sqrt(len(vals2)) if len(vals2) > 1 else np.nan)

        if len(vals1) > 1 and len(vals2) > 1:
            
            # # Using t-test:
            t, p = ttest_ind(vals1, vals2, equal_var=False, nan_policy='omit')
            
            # Using Mann Whitney U test:
            # u, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
            
            p_values.append(p)
        else:
            p_values.append(np.nan)

    if np.all(np.isnan(means_exp1)) and np.all(np.isnan(means_exp2)):
        print("No data available for the selected days and experiments.")
        return

    # Plot line graphs with SEM shading
    fig, ax = plt.subplots(figsize=(8,6), dpi=300)

    means_exp1 = np.array(means_exp1)
    sems_exp1 = np.array(sems_exp1)
    means_exp2 = np.array(means_exp2)
    sems_exp2 = np.array(sems_exp2)
    
    ax.plot(selected_days, means_exp1, '-o', color='black', label=exp1)
    ax.fill_between(selected_days, means_exp1 - sems_exp1, means_exp1 + sems_exp1,
                    color='black', alpha=0.2)

    ax.plot(selected_days, means_exp2, '-o', color='#5C3317', label=exp2)
    ax.fill_between(selected_days, means_exp2 - sems_exp2, means_exp2 + sems_exp2,
                    color='#5C3317', alpha=0.2)

    # Significance stars
    for x, y1, y2, p in zip(selected_days, means_exp1, means_exp2, p_values):
        if not np.isnan(p):
            if p < 0.001: stars = '***'
            elif p < 0.01: stars = '**'
            elif p < 0.05: stars = '*'
            else: stars = ''

            y_max = np.nanmax([y1, y2])
            ax.text(x, y_max + 0.05, stars, ha='center', va='bottom',
                    fontsize=16, color='black')

    # Add Chance line
    plt.hlines(0.5, xmin=selected_days[0], xmax=selected_days[-1], colors="grey", linestyles="--", label='Chance')

    ax.set_xlabel("Training day" if use_day_index else "Postnatal day", fontsize=16)
    ax.set_ylabel("Alternation rate", fontsize=16)
    ax.set_ylim(0,1.1)
    ax.set_title("Comparison of alternation rates", fontsize=17)
    ax.legend(loc='lower right', fontsize=15)
    ax.grid(alpha=0.6, linestyle='--')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Individual success rate', fontsize=16)
    
    # file_path = os.path.join(directory, f"graph_compare_{exp1}_vs_{exp2}.png")
    plt.tight_layout()
    # plt.savefig(file_path)
    plt.show()



from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_compare_experiments_graph(df, directory, exp1, exp2, selected_days,
                                   use_day_index=True):
    """
    Plots mean alternation rate for two experiment types with SEM shading.
    Performs Shapiro–Wilk normality tests, chooses t-test or Mann–Whitney U accordingly,
    and displays both significance stars and test-type indicators.
    Normal groups are marked with 'N' below the x-axis.
    """
    df = df.copy()
    df["experiment_type"] = df["experiment_type"].str.lower()
    exp1, exp2 = exp1.lower(), exp2.lower()

    x_col = "day_index" if use_day_index else "age"
    grouped = df.groupby(['experiment_type', x_col])['success_rate']

    means_exp1, sems_exp1, means_exp2, sems_exp2, p_values, test_labels = [], [], [], [], [], []
    normality_map = {exp1: [], exp2: []}

    for day in selected_days:
        try:
            vals1 = grouped.get_group((exp1, day)).dropna()
        except KeyError:
            vals1 = []
            print(f"No data for {exp1} on {x_col}={day}")
        try:
            vals2 = grouped.get_group((exp2, day)).dropna()
        except KeyError:
            vals2 = []
            print(f"No data for {exp2} on {x_col}={day}")
    
        # --- ⬇️ HER legger du inn visuell normalitetssjekk ---
        normal1 = check_normality_visual(vals1, exp1, day)
        normal2 = check_normality_visual(vals2, exp2, day)
        # ------------------------------------------------------
    
        # Her kan du så fortsette som før:
        if normal1 and normal2:
            t, p = ttest_ind(vals1, vals2, equal_var=False, nan_policy='omit')
            test_label = "T"
        else:
            t, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
            test_label = "U"
            test_labels.append(test_label)
            p_values.append(p)
            
            print(f"  {test_label}-test p={p:.4e}")
        # else:
        #     p_values.append(np.nan)
        #     test_labels.append("")
        #     print(f"  Not enough data for {day}")

    if np.all(np.isnan(means_exp1)) and np.all(np.isnan(means_exp2)):
        print("No data available for the selected days and experiments.")
        return

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    means_exp1, sems_exp1 = np.array(means_exp1), np.array(sems_exp1)
    means_exp2, sems_exp2 = np.array(means_exp2), np.array(sems_exp2)

    ax.plot(selected_days, means_exp1, '-o', color='black', label=exp1)
    ax.fill_between(selected_days, means_exp1 - sems_exp1, means_exp1 + sems_exp1,
                    color='black', alpha=0.2)

    ax.plot(selected_days, means_exp2, '-o', color='#5C3317', label=exp2)
    ax.fill_between(selected_days, means_exp2 - sems_exp2, means_exp2 + sems_exp2,
                    color='#5C3317', alpha=0.2)

    # Add significance stars + test labels
    for x, y1, y2, p, label in zip(selected_days, means_exp1, means_exp2, p_values, test_labels):
        if not np.isnan(p):
            if p < 0.001: stars = '***'
            elif p < 0.01: stars = '**'
            elif p < 0.05: stars = '*'
            else: stars = ''

            if stars:
                y_max = np.nanmax([y1, y2])
                ax.text(x, y_max + 0.05, stars, ha='center', va='bottom',
                        fontsize=16, color='black')
                # Add test-type label (T or U)
                ax.text(x, y_max + 0.085, label, ha='center', va='bottom',
                        fontsize=10, color='grey', fontweight='bold')

    # Mark normality below x-axis
    for exp, color in zip([exp1, exp2], ['black', '#5C3317']):
        for x, norm in zip(selected_days, normality_map[exp]):
            if norm:
                ax.text(x, -0.04, "", ha="center", va="top",
                        fontsize=11, color=color, fontweight="bold")

    # Add Chance line
    plt.hlines(0.5, xmin=selected_days[0], xmax=selected_days[-1],
               colors="grey", linestyles="--", label='Chance')

    ax.set_xlabel("Training day" if use_day_index else "Postnatal day", fontsize=16)
    ax.set_ylabel("Alternation rate", fontsize=16)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Comparison of alternation rates", fontsize=17)
    ax.legend(loc='lower right', fontsize=15)
    ax.grid(alpha=0.6, linestyle='--')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    
    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Individual success rate', fontsize=16)

    # file_path = os.path.join(directory, f"graph_compare_{exp1}_vs_{exp2}.png")
    # plt.savefig(file_path, dpi=300)
    plt.show()




def plot_compare_std_histogram(df, directory, experiments, selected_days, use_day_index=True, test="levene"):
    """
    Compare standard deviations of success_rate across multiple experiment types.
    Plots histogram of standard deviations with significance stars for pairwise comparisons.
    """
    x_col = "day_index" if use_day_index else "age"
    grouped = df.groupby(["experiment_type", x_col])["success_rate"]

    stds = {exp: [] for exp in experiments}
    for exp in experiments:
        for day in selected_days:
            try:
                vals = grouped.get_group((exp, day))
                stds[exp].append(np.std(vals, ddof=1) if len(vals) > 1 else np.nan) # find std
            except KeyError:
                stds[exp].append(np.nan)
                print(f"No data for {exp} on {x_col}={day}")

    # Plot
    x = np.arange(len(selected_days))
    width = 0.8 / len(experiments)  # spacing
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#006A4E", "#0041C2", "royalblue", "darkorange"]
    for i, exp in enumerate(experiments):
        ax.bar(x + i*width - (len(experiments)-1)*width/2,
                stds[exp], width,
                label=exp, color=colors[i % len(colors)], alpha=0.7)

    # Pairwise variance tests for each day
    for j, day in enumerate(selected_days):
        vals_by_exp = []
        for exp in experiments:
            try:
                vals = grouped.get_group((exp, day))
                vals_by_exp.append(vals)
            except KeyError:
                vals_by_exp.append([])

        # Compare all pairs
        for i in range(len(experiments)):
            for k in range(i+1, len(experiments)):
                if len(vals_by_exp[i]) > 1 and len(vals_by_exp[k]) > 1:
                    if test == "levene":
                        stat, p = levene(vals_by_exp[i], vals_by_exp[k])

                    if p < 0.001: stars = "***"
                    elif p < 0.01: stars = "**"
                    elif p < 0.05: stars = "*"
                    else: stars = ""

                    if stars:
                        y_max = np.nanmax([stds[experiments[i]][j], stds[experiments[k]][j]])
                        ax.text(j, y_max + 0.02, stars, ha="center", fontsize=14, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"P{d}" for d in selected_days], fontsize=18)
    ax.set_ylabel("Standard deviation of alternation rate", fontsize=18)
    ax.set_title(f"Comparison of standard deviations ({test.title()} test)", fontsize=20)
    ax.legend(fontsize=14)
    ax.grid(axis="y", alpha=0.6, linestyle="--")
    ax.set_ylim(0, max([np.nanmax(s) for s in stds.values()])*1.2)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # file_path = os.path.join(directory, f"std_histogram_compare_{'_vs_'.join(experiments)}.png")
    plt.tight_layout()
    # plt.savefig(file_path)
    plt.show()








import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, norm, probplot


from scipy.stats import shapiro, probplot
import matplotlib.pyplot as plt

def check_normality_visual(data, exp, day):
    """
    Performs Shapiro–Wilk test and visualizes the distribution with a Q–Q plot only.
    Returns True if data are normally distributed, else False.
    """
    if len(data) < 3:
        print(f"Not enough data for visual inspection ({exp}, day {day})")
        return False

    stat, p = shapiro(data)
    normal = p > 0.05

    print(f"\n{exp} – Day {day}: Shapiro p={p:.4f} → {'Normal' if normal else 'Non-normal'}")

    # --- Q–Q plot only ---
    plt.figure(figsize=(4, 4), dpi=200)
    probplot(data, dist="norm", plot=plt)
    plt.title(f"{exp} – Day {day}\nShapiro p={p:.4f} ({'Normal' if normal else 'Non-normal'})",
              fontsize=10)
    plt.tight_layout()
    plt.show()

    return normal



def plot_compare_experiments_graph(df, directory, exp1, exp2, selected_days, use_day_index=True):
    """
    Plots mean alternation rate for two experiment types with SEM shading.
    Performs Shapiro–Wilk normality tests (with visual inspection),
    chooses t-test or Mann–Whitney U accordingly,
    and displays both significance stars and test-type indicators.
    Normal groups are marked with 'N' below the x-axis.
    """
    df = df.copy()
    df["experiment_type"] = df["experiment_type"].str.lower()
    exp1, exp2 = exp1.lower(), exp2.lower()

    x_col = "day_index" if use_day_index else "age"
    grouped = df.groupby(['experiment_type', x_col])['success_rate']

    means_exp1, sems_exp1, means_exp2, sems_exp2 = [], [], [], []
    p_values, test_labels = [], []
    normality_map = {exp1: [], exp2: []}

    for day in selected_days:
        try:
            vals1 = grouped.get_group((exp1, day)).dropna()
        except KeyError:
            vals1 = []
            print(f"No data for {exp1} on {x_col}={day}")

        try:
            vals2 = grouped.get_group((exp2, day)).dropna()
        except KeyError:
            vals2 = []
            print(f"No data for {exp2} on {x_col}={day}")

        # Mean and SEM
        means_exp1.append(np.mean(vals1) if len(vals1) > 0 else np.nan)
        sems_exp1.append(np.std(vals1, ddof=1)/np.sqrt(len(vals1)) if len(vals1) > 1 else np.nan)
        means_exp2.append(np.mean(vals2) if len(vals2) > 0 else np.nan)
        sems_exp2.append(np.std(vals2, ddof=1)/np.sqrt(len(vals2)) if len(vals2) > 1 else np.nan)

        # --- ✅ Visuell normalitetssjekk (Shapiro + plott) ---
        normal1 = check_normality_visual(vals1, exp1, day)
        normal2 = check_normality_visual(vals2, exp2, day)
        normality_map[exp1].append(normal1)
        normality_map[exp2].append(normal2)
        # ------------------------------------------------------

        # --- Velg test basert på normalitet ---
        if len(vals1) > 1 and len(vals2) > 1:
            if normal1 and normal2:
                t, p = ttest_ind(vals1, vals2, equal_var=False, nan_policy='omit')
                test_label = "T"
            else:
                t, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
                test_label = "U"

            p_values.append(p)
            test_labels.append(test_label)
            print(f"Day {day}: {test_label}-test p={p:.4e}")
        else:
            p_values.append(np.nan)
            test_labels.append("")
            print(f"Day {day}: Not enough data")

    # --- Plot setup ---
    if np.all(np.isnan(means_exp1)) and np.all(np.isnan(means_exp2)):
        print("No data available for the selected days and experiments.")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    means_exp1, sems_exp1 = np.array(means_exp1), np.array(sems_exp1)
    means_exp2, sems_exp2 = np.array(means_exp2), np.array(sems_exp2)

    ax.plot(selected_days, means_exp1, '-o', color='black', label=exp1)
    ax.fill_between(selected_days, means_exp1 - sems_exp1, means_exp1 + sems_exp1,
                    color='black', alpha=0.2)

    ax.plot(selected_days, means_exp2, '-o', color='#5C3317', label=exp2)
    ax.fill_between(selected_days, means_exp2 - sems_exp2, means_exp2 + sems_exp2,
                    color='#5C3317', alpha=0.2)

    # --- Signifikansstjerner + testetikett ---
    for x, y1, y2, p, label in zip(selected_days, means_exp1, means_exp2, p_values, test_labels):
        if not np.isnan(p):
            if p < 0.001: stars = '***'
            elif p < 0.01: stars = '**'
            elif p < 0.05: stars = '*'
            else: stars = ''
            if stars:
                y_max = np.nanmax([y1, y2])
                ax.text(x, y_max + 0.05, stars, ha='center', va='bottom',
                        fontsize=16, color='black')
                ax.text(x, y_max + 0.085, label, ha='center', va='bottom',
                        fontsize=10, color='grey', fontweight='bold')

    # --- Marker normalitet under x-aksen ---
    for exp, color in zip([exp1, exp2], ['black', '#5C3317']):
        for x, norm in zip(selected_days, normality_map[exp]):
            if norm:
                ax.text(x, -0.04, "N", ha="center", va="top",
                        fontsize=11, color=color, fontweight="bold")

    # --- Chance line ---
    plt.hlines(0.5, xmin=selected_days[0], xmax=selected_days[-1],
               colors="grey", linestyles="--", label='Chance')

    ax.set_xlabel("Training day" if use_day_index else "Postnatal day", fontsize=16)
    ax.set_ylabel("Alternation rate", fontsize=16)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Comparison of alternation rates", fontsize=17)
    ax.legend(loc='lower right', fontsize=15)
    ax.grid(alpha=0.6, linestyle='--')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    
    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Comparison of success rates', fontsize=16)

    # file_path = os.path.join(directory, f"graph_compare_{exp1}_vs_{exp2}.png")
    # plt.savefig(file_path, dpi=300)
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import shapiro, ttest_ind, mannwhitneyu


# def check_normality_visual(data, exp, day):
#     """
#     Performs Shapiro–Wilk test and (optionally) visualizes the distribution with a Q–Q plot.
#     Returns True if data are normally distributed, else False.
#     """
#     if len(data) < 3:
#         return False
#     stat, p = shapiro(data)
#     normal = p > 0.05
#     print(f"{exp} – Day {day}: Shapiro p={p:.4f} → {'Normal' if normal else 'Non-normal'}")
#     return normal


def plot_compare_experiments_graph(df, directory, exp1, exp2, selected_days, use_day_index=True):
    """
    Plots mean alternation rate for two experiment types with SEM shading.
    Uses Shapiro–Wilk test to decide between t-test and Mann–Whitney U.
    Displays significance stars only (no N or T/U labels).
    """
    df = df.copy()
    df["experiment_type"] = df["experiment_type"].str.lower()
    exp1, exp2 = exp1.lower(), exp2.lower()

    x_col = "day_index" if use_day_index else "age"
    grouped = df.groupby(['experiment_type', x_col])['success_rate']

    means_exp1, sems_exp1, means_exp2, sems_exp2, p_values = [], [], [], [], []

    for day in selected_days:
        # Hent data
        vals1 = grouped.get_group((exp1, day)).dropna() if (exp1, day) in grouped.groups else []
        vals2 = grouped.get_group((exp2, day)).dropna() if (exp2, day) in grouped.groups else []

        # Beregn gjennomsnitt og SEM
        means_exp1.append(np.mean(vals1) if len(vals1) > 0 else np.nan)
        sems_exp1.append(np.std(vals1, ddof=1)/np.sqrt(len(vals1)) if len(vals1) > 1 else np.nan)
        means_exp2.append(np.mean(vals2) if len(vals2) > 0 else np.nan)
        sems_exp2.append(np.std(vals2, ddof=1)/np.sqrt(len(vals2)) if len(vals2) > 1 else np.nan)

        # Normalitet og valg av test
        if len(vals1) > 2 and len(vals2) > 2:
            normal1 = check_normality_visual(vals1, exp1, day)
            normal2 = check_normality_visual(vals2, exp2, day)
            if normal1 and normal2:
                _, p = ttest_ind(vals1, vals2, equal_var=False, nan_policy='omit')
            else:
                _, p = mannwhitneyu(vals1, vals2, alternative="two-sided")
        else:
            p = np.nan
        p_values.append(p)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    means_exp1, sems_exp1 = np.array(means_exp1), np.array(sems_exp1)
    means_exp2, sems_exp2 = np.array(means_exp2), np.array(sems_exp2)


    ax.plot(selected_days, means_exp1, '-o', color='#0041C2', label='P23-DNMP')
    ax.fill_between(selected_days, means_exp1 - sems_exp1, means_exp1 + sems_exp1,
                    color='#0041C2', alpha=0.2)
    ax.plot(selected_days, means_exp2, '-o', color='#006A4E', label='P15-DNMP')
    ax.fill_between(selected_days, means_exp2 - sems_exp2, means_exp2 + sems_exp2,
                    color='#006A4E', alpha=0.2)

    # Signifikansstjerner
    for x, y1, y2, p in zip(selected_days, means_exp1, means_exp2, p_values):
        if not np.isnan(p):
            if p < 0.001: stars = '***'
            elif p < 0.01: stars = '**'
            elif p < 0.05: stars = '*'
            else: stars = ''
            if stars:
                y_max = np.nanmax([y1, y2])
                ax.text(x, y_max + 0.05, stars, ha='center', va='bottom',
                        fontsize=16, color='black')

    # Chance-linje
    ax.hlines(0.5, xmin=selected_days[0], xmax=selected_days[-1],
              colors="grey", linestyles="--", label='Chance')

    ax.set_xlabel("Training day" if use_day_index else "Postnatal day", fontsize=16)
    ax.set_ylabel("Alternation rate", fontsize=16)
    ax.set_ylim(0, 1.1)
    ax.set_title("Comparison of alternation rates", fontsize=17)
    ax.legend(loc='lower right', fontsize=14)
    ax.grid(alpha=0.6, linestyle='--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    
    ax = plt.gca()
    # Remove top & right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thicken bottom & left spines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # success isf alternation
    ax.set_ylabel('Success rate', fontsize=16)
    ax.set_title(f'Comparison of success rates', fontsize=16)

    # Lagre figur
    # file_path = os.path.join(directory, f"graph_compare_{exp1}_vs_{exp2}.png")
    # plt.savefig(file_path, dpi=300)
    plt.show()







### EXECUTION ####

data_path = Path(r"C:\Users\solho\OneDrive - Universitetet i Oslo\Documents\Essentials\Data\alternation_with_time_all.xlsx") # excel file with ID, age, success rate, sex, experiment-type for all rats
ymaze_file = Path(r'C:\Users\solho\Desktop\info y maze wo empty.xlsx') # excel file with info about ID, age, experiment-type and result from each trial
rats_file = Path(r'C:\Users\solho\Desktop\info rats.xlsx') # info about rats (needed for getting info about the sex of animals)
directory = Path(r'C:\Users\solho\Desktop\y maze\alternation') # where to save figures
df = pd.read_excel(data_path)
df_all = pd.read_excel(ymaze_file)


# Decide which experiment types to analyze
# experiment_types = ['sd', 'spontaneous', 'experience', 'regular_DNMP']
experiment_types = ['sd']

for experiment_type in experiment_types:
    filtered_df = df[df["experiment_type"].str.lower() == experiment_type.lower()]
    
    # compute binomial sign. tresholds:
    if experiment_type == "spontaneous":
        threshold_by_day = compute_daily_significance_threshold(df_all, experiment_type, alpha=0.05)
        ages_sorted = sorted(threshold_by_day.keys())
        thresholds_sorted = {
            "lower_threshold": [threshold_by_day[age]["lower_threshold"] for age in ages_sorted],
            "upper_threshold": [threshold_by_day[age]["upper_threshold"] for age in ages_sorted]
        }
        
    else:
        threshold_by_day = compute_daily_significance_threshold(df_all, experiment_type, alpha=0.05)
        ages_sorted = sorted(threshold_by_day.keys())
        thresholds_sorted = [threshold_by_day[age] for age in ages_sorted] 
        
    # plot
    plot_individual_alternation(filtered_df, directory, experiment_type, ages_sorted, thresholds_sorted)
    plot_sex_mean_alternation(filtered_df, directory, experiment_type, ages_sorted, thresholds_sorted)
    plot_all_mean_alternation(filtered_df, directory, experiment_type, ages_sorted, thresholds_sorted)

    plot_regression_with_individuals_alternation(
        df=filtered_df,
        y_col="success_rate",
        ylabel="Alternation rate",
        directory=directory,
        experiment_type=experiment_type,
        filename_suffix="alternation_regression")


    plot_spearman_with_individuals_alternation(
        df=filtered_df,
        y_col="success_rate",
        ylabel="Alternation rate",
        directory=directory,
        experiment_type=experiment_type,
        filename_suffix="alternation_regression")
    
    
    

########### Compare experiments #########
# If you want to use day_index
df = df.sort_values(["experiment_type", "ID", "age"])
df["day_index"] = df.groupby(["experiment_type", "ID"]).cumcount() + 1

# choose experiments and days/ages to compare:
exp1 = "sd"
exp2 = "regular_DNMP"
# selected_days=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
# selected_days=[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30]
selected_days=[23,24,25,26,27,28,29,30]

plot_compare_experiments_graph(df, directory, exp1, exp2, selected_days, use_day_index=False)


## Compare std ###
# choose experiments and days/ages to compare:
exp_list = ["experience", "sd"]
days = [23,24,25,26,27,28,29,30]

plot_compare_std_histogram(df, directory, experiments=exp_list,
                           selected_days=days, use_day_index=False, test="levene")
