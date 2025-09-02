#%%Library Calls and Function Definition
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:15:27 2025

Last Edited: Tue Sep 2 2025

@author: Anthony Navarro

AI assistance was used for the purpose of streamlining, troubleshooting, or expediting the code writing process
AI was not used for the generation of ideas, pipelines, or data.
"""
#For ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (GroupShuffleSplit, StratifiedKFold)
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_curve)
from scipy.stats import sem, norm
from numpy import mean
#from AN_Biblioteca import *

def check_col(df):
    """
    Prints the percent missingness in every column

    Parameters:
    - df: dataset of interset

    Returns:
    - percent missingness in every column
    """
    missing_cols = []  # List to store column names
    total_rows = len(df)  # Get total number of rows
    
    for col in df.columns:
        missing_fraction = df[col].isnull().sum() / total_rows  # Compute missing percentage
        
        print(f'{col}: Missing = {missing_fraction:.2}')  # Display as percentage
        
        if missing_fraction > 0.33:  # If more than 33% missing
            missing_cols.append(col)  # Store the column name
    
    return missing_cols  # Return the list of columns

#For RL, not yet called in
from scipy.optimize import minimize
from matplotlib.lines import Line2D
import seaborn as sns
import statsmodels.api as sm

def conf_int(data, confidence= 0.95, return_range_only = True):
    """
    Calculate the confidence interval of the mean for a dataset.

    Parameters:
    - data: a set of values you wish to calculate the confidence interval for
    - confidence: Confidence level as a decimal (e.g., 0.95 for 95%)
    - return_range_only: If TRUE, return only the margin of error (half-width of the CI)

    Returns:
    - If return_range_only: float (margin of error)
    - Else: tuple (margin of error, upper CI, lower CI)
    """
    m = np.mean(data, axis=0)
    s = sem(data, axis=0)  # standard error of the mean (StD/(n^0.5))
    z = norm.ppf(0.5 + confidence / 2)  # z-score

    margin = z * s
    upper = m + margin
    lower = m - margin

    if return_range_only: #set to true by default for the purposes of this project
        return margin 
    else:
        return margin, upper, lower
    
def per_gain(final, initial, percent = False):
    """
    Calculate the percent change, used for gain in this script
    
    Parameters:
    - Final = Point to be compared
    - Initial = Comparsion Point
    - Percent = Whether you want to receive the value as a percent (0-100) or as a decimal (0-1)
    """
    pc = (final - initial)/initial
    if (percent == False): #Set to false for the purposes of this project
        return pc
    else:
        return pc*100

#%% Data Set-Up
sobreviver = pd.read_csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver.csv")
XandY = sobreviver.copy()
XandY = XandY[XandY['final_tt_credits'] >= 120]

XandY = XandY.drop(columns = ["cohort","Repeat_ID","tier1_STEM_drop_tozero", "semester_drop","max_semester_tally", #this is just for modeling reasons
                          "CSWS_Oa_score","CSWS_Ap_score","CSWS_Co_score","CSWS_Ac_score","CSWS_Fa_score","CSWS_Vt_score","CSWS_Gl_score","MSPSS_SO_score","MSPSS_Fa_score",
                          "MSPSS_Fr_score","SIR_DS_score","SIR_Aq_score","EPSI_Pg_score","EPSI_EE_score","EPSI_Rt_score","EPSI_BE_score","final_UM_credits","percent_complete_tt_crd",
                          "final_tt_credits","percent_complete_UM_crd",
                          'crd_load_median','Current_average_mean','Current_average_sd','delta_GPA_sum', 'delta_GPA_median', 'delta_GPA_sd',"final_GPA",
                          "percent_complete_sem"]) 
#All of these columns were removed due to either being subscores of a scale or having information that would NOT be available during the study semester

check_col(XandY)
XandY = XandY.dropna()
print("Sample Size:", len(XandY))
print("Pre-Dropna Sample Size:", len(sobreviver))
print("Ratio of Post-Drop/Pre-Drop", len(XandY)/len(sobreviver))

X = XandY.copy()

X = X.drop(columns=["final_major_t1_STEM"]) # Keep ID for now


y = XandY.copy()
y = XandY["final_major_t1_STEM"]
#%% Simple Logistic Regresion Using only Demograhic Data

#Plots for this aren't vital, reporting these statistic will suffice. For calulating gain, we can just do (AUC_RFC - AUC_LR)/(AUC_LR)

Demograph = ["gender_F","race_MLT","race_CWH","race_AFB","race_AAA","race_NAT","race_HPI","race_OTH","ethnicity_HS"]

X = XandY.copy()
X = X[Demograph]

y = XandY.copy()
y = XandY["final_major_t1_STEM"]

# Configuration
n_splits = 5
n_repeats = 100

# Storage
roc_curves = []
auc_scores = []
accuracies = []
precisions = []
recalls = []
sensitivities = []
specificities = []
f1_scores = []
conf_matrices = []
feature_rows_perm = []


# Coefficient storage to summarize later (per feature)
feature_names = list(X.columns)
coef_history = {f: [] for f in feature_names}
intercept_history = []

# --- CV Loop ---
for seed in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        logit = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            class_weight=None,
            random_state=seed
        )
        logit.fit(X_train, y_train)

        # Predictions
        y_pred = logit.predict(X_test)
        y_proba = logit.predict_proba(X_test)[:, 1]

        # Metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivities.append(tp / (tp + fn) if (tp + fn) else np.nan)  # recall
        specificities.append(tn / (tn + fp) if (tn + fp) else np.nan)
        conf_matrices.append(np.array([[tn, fp], [fn, tp]]))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append((fpr, tpr))

        # Coefficients
        coefs = pd.Series(logit.coef_.ravel(), index=feature_names)
        for f in feature_names:
            coef_history[f].append(coefs[f])
        intercept_history.append(float(logit.intercept_[0]))

# --- Aggregate metrics ---
metric_summary_LR = {
    "Accuracy": (mean(accuracies), conf_int(accuracies)),
    "ROC AUC": (mean(auc_scores), conf_int(auc_scores)),
    "Precision": (mean(precisions), conf_int(precisions)),
    "Recall (Sensitivity)": (mean(recalls), conf_int(recalls)),
    "Specificity": (mean(specificities), conf_int(specificities)),
    "F1 Score": (mean(f1_scores), conf_int(f1_scores))
}

# Mean confusion matrix and CI 
mean_conf_matrix = np.mean(conf_matrices, axis=0)
conf_matrix_ci = conf_int(conf_matrices)

# --- Coefficient & Odds Ratio summaries with CIs ---
coef_summary_rows = []
for f in feature_names:
    vals = coef_history[f]
    # coefficient CI
    margin, upper, lower = conf_int(vals, return_range_only=False)
    mean_coef = mean(vals)
    # odds ratio CI
    or_mean  = float(np.exp(mean_coef))
    or_lower = float(np.exp(lower))
    or_upper = float(np.exp(upper))
    coef_summary_rows.append({
        "feature": f,
        "coef_mean": mean_coef,
        "coef_CI_lower": lower,
        "coef_CI_upper": upper,
        "OR_mean": or_mean,
        "OR_CI_lower": or_lower,
        "OR_CI_upper": or_upper
    })
coef_summary = pd.DataFrame(coef_summary_rows).set_index("feature").sort_values("OR_mean", ascending=False)

# Outputs
print("Logistic Regression Metric Summary (mean, CI):")
for k, (m, ci) in metric_summary_LR.items():
    print(f"{k:>18}: {m:.3f} ({m-ci:.3f}, {m+ci:.3f})")

print("\nMean Confusion Matrix over folds/runs:")
print(mean_conf_matrix.round(2), "\n\n(LCI:\n", mean_conf_matrix.round(2) - conf_matrix_ci.round(2), "\n,\nUCI:\n", mean_conf_matrix.round(2) + conf_matrix_ci.round(2) ,")")

print("\nCoefficient / Odds Ratio summary:")
with pd.option_context('display.max_columns', None):
    print(coef_summary)


#%% Pre-Pruning Base Model

X = XandY.copy()
X = X.drop(columns=["final_major_t1_STEM"]) # Keep ID for now

y = XandY.copy()
y = XandY["final_major_t1_STEM"]

# Configuration
n_splits = 5
n_repeats = 100

# Storage
roc_curves = []
auc_scores = []
accuracies = []
precisions = []
recalls = []
sensitivities = []
specificities = []
f1_scores = []
conf_matrices = []
feature_rows = []

for seed in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        feature_names = list(X_train.columns)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        conf_matrices.append(confusion_matrix(y_test, y_pred))
        
        # Sensitivity and Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivities.append(tp / (tp + fn))
        specificities.append(tn / (tn + fp))
        
        conf_matrices.append(np.array([[tn, fp], [fn, tp]]))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append((fpr, tpr))

        # Feature importances
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1)
        importances = pd.Series(result.importances_mean, index=feature_names)
        stds = pd.Series(result.importances_std, index=feature_names)

        importance_row = {
            col: imp if (imp - std > 0 or imp + std < 0) else np.nan
            for col, imp, std in zip(feature_names, importances, stds)
        }
        feature_rows.append(importance_row)

# Aggregate metrics
metric_summary_ppRFC = {
    "Accuracy": (mean(accuracies), conf_int(accuracies)),
    "ROC AUC": (mean(auc_scores), conf_int(auc_scores)),
    "Precision": (mean(precisions), conf_int(precisions)),
    "Recall (Sensitivity)": (mean(recalls), conf_int(recalls)),
    "Specificity": (mean(specificities), conf_int(specificities)),
    "F1 Score": (mean(f1_scores), conf_int(f1_scores))
}

# Mean confusion matrix
mean_conf_matrix = np.mean(conf_matrices, axis=0)
conf_matrix_ci = conf_int(conf_matrices)

#%% Pre-Pruning Plots

### Plot ROC AUC ###

# Common FPR grid for interpolation
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []

# Interpolate all TPRs onto mean FPR grid
for fpr, tpr in roc_curves:
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

# Convert to numpy array
interp_tprs = np.array(interp_tprs)

# Compute mean and 95CI
mean_tpr = np.mean(interp_tprs, axis=0)
CI_tpr = conf_int(accuracies)#(interp_tprs, axis=0)
mean_tpr[-1] = 1.0

# Upper and lower bounds
tpr_upper = np.minimum(mean_tpr + CI_tpr, 1)
tpr_lower = np.maximum(mean_tpr - CI_tpr, 0)

# Mean AUC
mean_auc = np.mean(auc_scores)

# All ROC curves
print(f"Number of individual ROC curves plotted: {len(roc_curves)}")

mean_auc = np.mean(auc_scores)
CI_auc = conf_int(auc_scores)
lower_auc = mean_auc - CI_auc
upper_auc = mean_auc + CI_auc

auc_label = f"Mean ROC (AUC = {mean_auc:.3f} [{lower_auc:.3f}, {upper_auc:.3f}])"

for fpr, tpr in roc_curves:
    plt.plot(fpr, tpr, alpha=0.1, color='gray')

# Mean curve
plt.plot(mean_fpr, mean_tpr, color='blue', label=auc_label, linewidth=2)

# 95CI band
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=.8, label="95% Confidence Interval")

plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)

# Labels and layout
plt.title("Mean ROC Curve For RFC run on Pre-Pruned Data ± 95% CI", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/100roc_curve_pprune.svg", format='svg')  # Save in vector format
plt.show()

# Create labels for mean ± 95CI
labels = np.empty_like(mean_conf_matrix, dtype=object)
for i in range(2):
    for j in range(2):
        mean_val = mean_conf_matrix[i, j]
        ci_val = conf_matrix_ci[i, j]
        labels[i, j] = f"{mean_val:.1f}\n({mean_val-ci_val:.1f}, {mean_val+ci_val:.1f})"

fig, ax = plt.subplots(figsize=(6, 5))
# Coordinates for pcolormesh (requires edges, so +1 shape)
x = np.arange(3)
y = np.arange(3)

# Plot confusion matrix as vector-based pcolormesh
c = ax.pcolormesh(x, y, mean_conf_matrix, cmap='Blues', shading='auto')

# Add colorbar
fig.colorbar(c, ax=ax)

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.5, labels[i, j], ha='center', va='center', color='black', fontsize=12)

# Axes settings
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(['NOT STEM Major', 'STEM Major'])
ax.set_yticklabels(['NOT STEM Major', 'STEM Major'], rotation=90)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Mean Confusion Matrix ± 95% CI (100×5 RFC, Pre-Pruning)", fontsize=14)

# Clean look: turn off spines and grid
ax.grid(False)
# for spine in ax.spines.values():
#     spine.set_visible(False)

plt.tight_layout()

# Save as SVG (should have no PNGs)
plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/mean_confusion_matrix_pprune.svg", format='svg')

plt.show()

print("=== Model Performance Metrics (mean ± 95 CI) ===")
for metric, (m, CI) in metric_summary_ppRFC.items():
    lci = m - CI
    uci = m + CI
    print(f"{metric}: {m:.3f} ({lci:.3f}, {uci:.3f})")
    
### Plot Feature Importances ###

importance_df = pd.DataFrame(feature_rows)

mean_importance = importance_df.mean(skipna=True)
std_importance = importance_df.std(skipna=True)

# Sort by importance
mean_importance = mean_importance.sort_values(ascending=False)
std_importance = std_importance[mean_importance.index]


# Set up figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Bar plot with error bars
mean_importance.plot.bar(
    yerr=std_importance,
    capsize=4,
    color='steelblue',
    edgecolor='black',
    ax=ax
)
# Labels and title
ax.set_title("Mean Feature Importance of Pre-Pruned Feature Set", fontsize=14)
ax.set_ylabel("Mean Permutation Importance", fontsize=12)
ax.set_xticklabels(mean_importance.index, rotation=45, ha='right', fontsize=10)
ax.tick_params(axis='y', labelsize=10)

# Grid behind bars
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', linewidth=0.5)

# Emphasize y=0 line
ax.axhline(0, color='black', linewidth=1.2)

# Layout and save
plt.tight_layout()
plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/feature_importance_pprune.svg", format='svg')
plt.show()


#%% Pruning [Don't run if you don't absolutely have to, takes at least 2 hours]


# --- CONFIGURATION ---
target_feature_count = 10
roc_auc_threshold_drop = 0.05
num_runs_per_step = 100
n_repeats_permutation = 10

# --- INITIAL SETUP ---
groups = XandY["ID"]
#y = XandY["final_major_t1_STEM"]
#X = X.drop(columns=["final_major_t1_STEM"])  # Remove target if present

current_features = [col for col in X.columns if col != "ID"]
removed_features = []
roc_auc_history = []
remaining_features_history = []

# Track metrics across pruning steps
pruning_stats = []

# --- PRUNING LOOP ---
while len(current_features) > target_feature_count:
    auc_scores = []
    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    specificity_scores = []
    importances_list = []

    for i in range(num_runs_per_step):
        random_state = random.randint(1, 10000)
        splitter = GroupShuffleSplit(test_size=0.5, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train = X.iloc[train_idx][current_features]
        X_test = X.iloc[test_idx][current_features]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = RandomForestClassifier(n_jobs=-1, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred))
        rec_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity_scores.append(tn / (tn + fp))

        # Permutation importance
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats_permutation,
            random_state=random_state,
            n_jobs=-1
        )
        importances_list.append(result.importances_mean)

    # Aggregate results
    mean_auc = np.mean(auc_scores)
    mean_acc = np.mean(acc_scores)
    mean_prec = np.mean(prec_scores)
    mean_rec = np.mean(rec_scores)
    mean_f1 = np.mean(f1_scores)
    mean_spec = np.mean(specificity_scores)

    roc_auc_history.append(mean_auc)
    remaining_features_history.append(len(current_features))

    if len(roc_auc_history) > 1:
        delta_auc = roc_auc_history[-2] - mean_auc
        if delta_auc > roc_auc_threshold_drop:
            print(f"\nStopping: AUC dropped by {delta_auc:.4f} (> {roc_auc_threshold_drop})")
            break

    importances_matrix = np.vstack(importances_list)
    mean_importances = np.mean(importances_matrix, axis=0)
    feature_series = pd.Series(mean_importances, index=current_features)

    least_important = feature_series.idxmin()
    removed_features.append(least_important)
    removed_importance = feature_series[least_important]
    current_features.remove(least_important)

    # Store pruning step stats
    pruning_stats.append({
        "Features_Remaining": len(current_features),
        "Dropped_Feature": least_important,
        "Dropped_Feature_Importance": removed_importance,
        "ROC_AUC": mean_auc,
        "Accuracy": mean_acc,
        "Precision": mean_prec,
        "Recall/Sensitivity": mean_rec,
        "Specificity": mean_spec,
        "F1_Score": mean_f1
    })

    print(f"Dropped: {least_important} | Mean AUC: {mean_auc:.4f} | Features left: {len(current_features)}")

# --- FINAL REPORT ---
print("\n=== FINAL RESULTS ===")
print(f"Final features ({len(current_features)}): {current_features}")
print(f"Features dropped ({len(removed_features)}): {removed_features}")
print(f"Final mean AUC: {roc_auc_history[-1]:.4f}")

# --- CONVERT TO DATAFRAME ---
pruning_df = pd.DataFrame(pruning_stats)
#%% Run Base Model (Post-Pruning)

#now take the remaining features and run 100 models using only those 10
current_features = ['extraversion', 'open_mindedness', 'term_GPA', 'UM_credits_at_study', 'semester_study', 'Goal_grade_sd', 'grade_100_mean', 'grade_minus_goal_mean', 'NA_dense_mean', 'pred_100_mean']


X = XandY.copy()
X = X[current_features]

y = XandY.copy()
y = XandY["final_major_t1_STEM"]



# Configuration
n_splits = 5
n_repeats = 100

# Storage
roc_curves = []
auc_scores = []
accuracies = []
precisions = []
recalls = []
sensitivities = []
specificities = []
f1_scores = []
conf_matrices = []
feature_rows = []

for seed in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        feature_names = list(X_train.columns)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        conf_matrices.append(confusion_matrix(y_test, y_pred))
        
        # Sensitivity and Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivities.append(tp / (tp + fn))
        specificities.append(tn / (tn + fp))
        
        conf_matrices.append(np.array([[tn, fp], [fn, tp]]))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append((fpr, tpr))

        # Feature importances
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1)
        importances = pd.Series(result.importances_mean, index=feature_names)
        stds = pd.Series(result.importances_std, index=feature_names)

        importance_row = {
            col: imp if (imp - std > 0 or imp + std < 0) else np.nan
            for col, imp, std in zip(feature_names, importances, stds)
        }
        feature_rows.append(importance_row)

# Aggregate metrics
metric_summary_pRFC = {
    "Accuracy": (mean(accuracies), conf_int(accuracies)),
    "ROC AUC": (mean(auc_scores), conf_int(auc_scores)),
    "Precision": (mean(precisions), conf_int(precisions)),
    "Recall (Sensitivity)": (mean(recalls), conf_int(recalls)),
    "Specificity": (mean(specificities), conf_int(specificities)),
    "F1 Score": (mean(f1_scores), conf_int(f1_scores))
}

# Mean confusion matrix
mean_conf_matrix = np.mean(conf_matrices, axis=0)
conf_matrix_ci = conf_int(conf_matrices)
#%% Plot Figures (Post-Pruning)

### Plot ROC AUC ###

# Common FPR grid for interpolation
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []

# Interpolate all TPRs onto mean FPR grid
for fpr, tpr in roc_curves:
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

# Convert to numpy array
interp_tprs = np.array(interp_tprs)

# Compute mean and 95CI
mean_tpr = np.mean(interp_tprs, axis=0)
CI_tpr = conf_int(accuracies)#(interp_tprs, axis=0)
mean_tpr[-1] = 1.0

# Upper and lower bounds
tpr_upper = np.minimum(mean_tpr + CI_tpr, 1)
tpr_lower = np.maximum(mean_tpr - CI_tpr, 0)

# Mean AUC
mean_auc = np.mean(auc_scores)

# All ROC curves
print(f"Number of individual ROC curves plotted: {len(roc_curves)}")

mean_auc = np.mean(auc_scores)
CI_auc = conf_int(auc_scores)
lower_auc = mean_auc - CI_auc
upper_auc = mean_auc + CI_auc

auc_label = f"Mean ROC (AUC = {mean_auc:.3f} [{lower_auc:.3f}, {upper_auc:.3f}])"

for fpr, tpr in roc_curves:
    plt.plot(fpr, tpr, alpha=0.1, color='gray')

# Mean curve
plt.plot(mean_fpr, mean_tpr, color='blue', label=auc_label, linewidth=2)

# 95CI band
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=.8, label="95% Confidence Interval")

plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)

# Labels and layout
plt.title("Mean ROC Curve For RFC run on Pruned Data ± 95% CI", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/100roc_curve_CI.svg", format='svg')  # Save in vector format
plt.show()

### Plot Confusion Matrix + Print Model Performance ###


# Create labels for mean ± 95CI
labels = np.empty_like(mean_conf_matrix, dtype=object)
for i in range(2):
    for j in range(2):
        mean_val = mean_conf_matrix[i, j]
        ci_val = conf_matrix_ci[i, j]
        labels[i, j] = f"{mean_val:.1f}\n({mean_val-ci_val:.1f}, {mean_val+ci_val:.1f})"

fig, ax = plt.subplots(figsize=(6, 5))
# Coordinates for pcolormesh (requires edges, so +1 shape)
x = np.arange(3)
y = np.arange(3)

# Plot confusion matrix as vector-based pcolormesh
c = ax.pcolormesh(x, y, mean_conf_matrix, cmap='Blues', shading='auto')

# Add colorbar
fig.colorbar(c, ax=ax)

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.5, labels[i, j], ha='center', va='center', color='black', fontsize=12)

# Axes settings
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(['NOT STEM Major', 'STEM Major'])
ax.set_yticklabels(['NOT STEM Major', 'STEM Major'], rotation=90)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Mean Confusion Matrix ± 95% CI (100×5 RFC, Pre-Pruning)", fontsize=14)

# Clean look: turn off spines and grid
ax.grid(False)
# for spine in ax.spines.values():
#     spine.set_visible(False)

plt.tight_layout()

# Save as SVG (should have no PNGs)
plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/mean_confusion_matrix_CI.svg", format='svg')

plt.show()

print("=== Model Performance Metrics (mean ± 95 CI) ===")
for metric, (m, CI) in metric_summary_pRFC.items():
    lci = m - CI
    uci = m + CI
    print(f"{metric}: {m:.3f} ({lci:.3f}, {uci:.3f})")
    
### Plot Feature Importances ###

importance_df = pd.DataFrame(feature_rows)

mean_importance = importance_df.mean(skipna=True)
std_importance = importance_df.std(skipna=True)

# Sort by importance
mean_importance = mean_importance.sort_values(ascending=False)
std_importance = std_importance[mean_importance.index]


# Set up figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Bar plot with error bars
mean_importance.plot.bar(
    yerr=std_importance,
    capsize=4,
    color='steelblue',
    edgecolor='black',
    ax=ax
)
# Labels and title
ax.set_title("Mean Feature Importance of Pruned Feature Set", fontsize=14)
ax.set_ylabel("Mean Permutation Importance", fontsize=12)
ax.set_xticklabels(mean_importance.index, rotation=45, ha='right', fontsize=10)
ax.tick_params(axis='y', labelsize=10)

# Grid behind bars
ax.set_axisbelow(True)
ax.grid(axis='y', linestyle='--', linewidth=0.5)

# Emphasize y=0 line
ax.axhline(0, color='black', linewidth=1.2)

# Layout and save
plt.tight_layout()
plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/feature_importance_CI.svg", format='svg')
plt.show()

#%% Calculate and Print Gain

print("\nRelative to the LR model, the Unpruned RFC had an improved ROC_AUC of",
      per_gain(metric_summary_ppRFC["ROC AUC"][0], metric_summary_LR["ROC AUC"][0]), "and an improved accuracy of",
      per_gain(metric_summary_ppRFC["Accuracy"][0], metric_summary_LR["Accuracy"][0]))

print("\nRelative to the LR model, the Pruned RFC had an improved ROC_AUC of",
      per_gain(metric_summary_pRFC["ROC AUC"][0], metric_summary_LR["ROC AUC"][0]), "and an improved accuracy of",
      per_gain(metric_summary_pRFC["Accuracy"][0], metric_summary_LR["Accuracy"][0]))

print("\nRelative to the Unpruned RFC, the Pruned RFC had an improved ROC_AUC of",
      per_gain(metric_summary_pRFC["ROC AUC"][0], metric_summary_ppRFC["ROC AUC"][0]), "and an improved accuracy of",
      per_gain(metric_summary_pRFC["Accuracy"][0], metric_summary_ppRFC["Accuracy"][0]))


#%%Reinforcement Learning -----------------------------------------------------
#%% Pre-process
sobreviver_RL = pd.read_csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver_II.csv")
XandY_RL = sobreviver_RL.copy()
XandY_RL = XandY_RL[XandY_RL['final_tt_credits'] >= 120]
pre_drop = XandY_RL.copy()

#This pares down the dataset to only relevant variables.
RL_model = ["ID","final_major_t1_STEM","tier1_STEM_drop_tozero",
            "grade_100_exam1", 'grade_100_exam2', 'grade_100_exam3','grade_100_exam4',
            "PE_exam1","PE_exam2","PE_exam3","PE_exam4",
            "pred_100_exam1", "pred_100_exam2", "pred_100_exam3", "pred_100_exam4",
            "Goal_grade_exam1", "Goal_grade_exam2", "Goal_grade_exam3", "Goal_grade_exam4",
            "grade_minus_goal_exam1", "grade_minus_goal_exam2", "grade_minus_goal_exam3", "grade_minus_goal_exam4"]
XandY_RL = XandY_RL[RL_model]


check_col(XandY_RL)

XandY_RL = XandY_RL.dropna()
print("Sample Size:", len(XandY_RL))
print("Pre-Dropna Sample Size:", len(pre_drop))
print("Ratio of Post-Drop/Pre-Drop", len(XandY_RL)/len(pre_drop))

#%% Reformat data prior to model

# Reshape data for PyMC model
long_data = []

for idx, row in XandY_RL.iterrows():
    subject_id = row['ID']
    final_stem = row['final_major_t1_STEM']
    dtz = row["tier1_STEM_drop_tozero"]
    
    for t in range(1, 4):  # Exams 1 to 3 only, since we need Goal_{t+1}
        long_data.append({
            'ID': subject_id,
            't': t,
            
            'goal_t': row[f'Goal_grade_exam{t}'],
            'goal_tp1': row[f'Goal_grade_exam{t+1}'],
            'pred_t': row[f'pred_100_exam{t}'],
            'pred_tp1': row[f'pred_100_exam{t+1}'],
            
            'PE_t': row[f'PE_exam{t}'],
            'grade_minus_goal_t': row[f'grade_minus_goal_exam{t}'],
            'STEM': final_stem,
            "dtz": dtz
            
        })

df_long = pd.DataFrame(long_data)
print(df_long.head())

#%% Run RL model

subject_alphas = []
subject_ids = df_long['ID'].unique()

for subj in subject_ids:
    subj_data = df_long[df_long['ID'] == subj]

    # Fit alpha_PE
    def loss_pe(alpha):
        pred = subj_data['goal_t'].values + alpha * subj_data['PE_t'].values
        error = pred - subj_data['goal_tp1'].values
        return np.sum(error**2)

    result_pe = minimize(loss_pe, x0=0.1, bounds=[(-1, 1)])
    alpha_pe = result_pe.x[0]

    # Fit alpha_GMG
    def loss_gmg(alpha):
        pred = subj_data['goal_t'].values + alpha * subj_data['grade_minus_goal_t'].values
        error = pred - subj_data['goal_tp1'].values
        return np.sum(error**2)

    result_gmg = minimize(loss_gmg, x0=0.1, bounds=[(-1, 1)])
    alpha_gmg = result_gmg.x[0]

    # Store results
    subject_alphas.append({
        'ID': subj,
        'alpha_PE': alpha_pe,
        'alpha_GMG': alpha_gmg,
        'STEM': subj_data['STEM'].iloc[0],
        'dtz': subj_data['dtz'].iloc[0] if 'dtz' in subj_data.columns else None
    })

df_alphas = pd.DataFrame(subject_alphas)
print(df_alphas.head())

#%% Plot + Save Indvidual Alphas
# Map STEM codes to labels
df_alphas['STEM_label'] = df_alphas['STEM'].map({1: 'Final Major: STEM', 0: 'Final Major: Non-STEM'})
df_alphas['dtz_label']  = df_alphas['dtz'].map({1: 'Dropped STEM Major', 0: 'Did Not Drop STEM Major'})

#Make sure to select which of GMG or PE you want to plot and which of DTZ and Final Maj you want to group by
#(**) = use this line if you want x = alpha_PE
#(\/) = use this line if you want x = alpha_GMG
#([]) = use this line if you want grouping = final major
#({}) = use this line if you want grouping = tier1_dtz

plt.figure(figsize=(10,6))
ax = sns.histplot(
    data=df_alphas, 
    x='alpha_PE', #(**) 
    #x='alpha_GMG', #(\/)
    hue='STEM_label', #([])
    # hue='dtz_label', #({})
    kde=True, 
    bins=30, 
    element='step',
    palette={'Final Major: STEM': 'blue', 'Final Major: Non-STEM': 'red'}, #([])
    #palette={'Did Not Drop STEM Major': 'blue', 'Dropped STEM Major': 'red'}, #({})
    legend=False  # hide default legend so we can customize it
)

# Add alpha = 0 line
plt.axvline(0, color='black', linestyle='--')

# Manually define legend
custom_lines = [
    Line2D([0], [0], color='blue', lw=2, label='Final Major: STEM'), Line2D([0], [0], color='red', lw=2, label='Final Major: Non-STEM'), #([])
    # Line2D([0], [0], color='blue', lw=2, label='Did Not Drop STEM Major'), Line2D([0], [0], color='red', lw=2, label='Dropped STEM Major'), #({})
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='α = 0')
]
plt.legend(handles=custom_lines, title='Group')

# Labels
plt.title('Distribution of Estimated Learning Rates (α) by Final STEM Outcome \nGoal(t+1) = (a * PE) + Goal(t)') #(**)
# plt.title('Distribution of Estimated Learning Rates (α) by Final STEM Outcome \nGoal(t+1) = (a * GMG) + Goal(t)') #(\/)
plt.xlabel('Learning Rate (α)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#Commented out only for the purpose of not rewriting data
# sobreviver_ori_1 = pd.read_csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver.csv")

# sobreviver_alpha_1 = sobreviver_ori_1.merge(df_alphas[['ID', 'alpha_PE', 'alpha_GMG']], on='ID', how='left')

# sobreviver_alpha_1.to_csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver_alpha.csv", index = False)

# sobreviver_ori_2 = pd.read_csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver_II.csv")

# sobreviver_alpha_2 = sobreviver_ori_2.merge(df_alphas[['ID', 'alpha_PE', 'alpha_GMG']], on='ID', how='left')

# sobreviver_alpha_2.to_csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver_II_alpha.csv", index = False)
#%% Fit global alpha_pos and alpha_neg

#Remember alpha_pos here means the GLOBAL sensitivity when PE_t > 0
#         alpha_neg here means the GLOBAL sensitivity when PE_t < 0
df_long['PE_pos'] = df_long['PE_t'].apply(lambda x: x if x > 0 else 0)
df_long['PE_neg'] = df_long['PE_t'].apply(lambda x: x if x < 0 else 0)

# Define outcome and predictors
X = df_long[['PE_pos', 'PE_neg']]
X = sm.add_constant(X)  # Adds intercept term
y = df_long['goal_tp1']

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary())


# Define loss function over all subjects
def pooled_loss(alphas):
    alpha_pos, alpha_neg = alphas
    pred = df_long['goal_t'].values + alpha_pos * df_long['PE_pos'].values + alpha_neg * df_long['PE_neg'].values
    error = pred - df_long['goal_tp1'].values
    return np.sum(error**2)

# Initial guess and bounds
x0 = [0.1, 0.1]
bounds = [(-1, 1), (-1, 1)]

# Minimize loss
result = minimize(pooled_loss, x0=x0, bounds=bounds)
alpha_pos, alpha_neg = result.x

print(f"\nEstimated α_pos (PE > 0): {alpha_pos:.4f}")
print(f"Estimated α_neg (PE < 0): {alpha_neg:.4f}")
print(f"Final loss: {result.fun:.4f}")
