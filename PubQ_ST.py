#%%Library Calls and Function Definition
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:15:27 2025

Last Edited: Mon Nov 3 2025

@author: Anthony Navarro

AI assistance was used solely for the purpose of streamlining and troubleshooting during the code writing process.
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

#for supplementary correlation matrix, not already called in
import seaborn as sns
from matplotlib.patches import Patch




#%% Data Set-Up
sobreviver = pd.read_csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver.csv")
XandY = sobreviver.copy()
XandY = XandY[XandY['final_tt_credits'] >= 120]

XandY = XandY.drop(columns = ["cohort","Repeat_ID", "semester_drop","max_semester_tally", #this is just for modeling reasons
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

X = X.drop(columns=["final_major_t1_STEM", "tier1_STEM_drop_tozero"]) # Keep ID for now
#it is vital that tier1_dtz is removed, it WILL cause overfitting if included and
#invalidate everything due to it being HIGHLY correlated with Final Major


y = XandY.copy()
y = XandY["final_major_t1_STEM"]
#%% Simple Logistic Regresion Using only Demograhic Data

#For calulating gain, we can just do (AUC_RFC - AUC_LR)/(AUC_LR)

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
# print("Logistic Regression Metric Summary (mean, CI):")
for k, (m, ci) in metric_summary_LR.items():
    print(f"{k:>18}: {m:.3f} ({m-ci:.3f}, {m+ci:.3f})")

print("\nCoefficient / Odds Ratio summary:")
with pd.option_context('display.max_columns', None):
    print(coef_summary)
#%% LRM Plotting Baseline
### Plot ROC AUC for Logistic Regression ###

# Common FPR grid for interpolation
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []

# Interpolate all TPRs onto mean FPR grid
for fpr, tpr in roc_curves:
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

interp_tprs = np.array(interp_tprs)
mean_tpr = np.mean(interp_tprs, axis=0)
CI_tpr = conf_int(accuracies)  # Used for plotting 95% CI band width
mean_tpr[-1] = 1.0

# Confidence bounds
tpr_upper = np.minimum(mean_tpr + CI_tpr, 1)
tpr_lower = np.maximum(mean_tpr - CI_tpr, 0)

# Mean AUC and CI
mean_auc = np.mean(auc_scores)
CI_auc = conf_int(auc_scores)
lower_auc = mean_auc - CI_auc
upper_auc = mean_auc + CI_auc

auc_label = f"Mean ROC (AUC = {mean_auc:.3f} [{lower_auc:.3f}, {upper_auc:.3f}])"

# Plot all ROC curves
print(f"Number of individual ROC curves plotted: {len(roc_curves)}")
for fpr, tpr in roc_curves:
    plt.plot(fpr, tpr, alpha=0.1, color='gray')

# Mean ROC
plt.plot(mean_fpr, mean_tpr, color='blue', label=auc_label, linewidth=2)
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=.3, label="95% Confidence Interval")
plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)

plt.title("ROC Curve for Logistic Regression Model", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/LRM_roc_curve.svg", format='svg')
plt.show()


### Plot Mean Confusion Matrix ###

# Create labels for mean ± 95CI
labels = np.empty_like(mean_conf_matrix, dtype=object)
for i in range(2):
    for j in range(2):
        mean_val = mean_conf_matrix[i, j]
        ci_val = conf_matrix_ci[i, j]
        labels[i, j] = f"{mean_val:.1f}\n({mean_val - ci_val:.1f}, {mean_val + ci_val:.1f})"

fig, ax = plt.subplots(figsize=(6, 5))

# Create grid
x = np.arange(3)
y = np.arange(3)

# Plot confusion matrix
c = ax.pcolormesh(x, y, mean_conf_matrix, cmap='Blues', shading='auto')
fig.colorbar(c, ax=ax)

# Add annotations
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.5, labels[i, j], ha='center', va='center', color='black', fontsize=12)

# Axis formatting
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(['NOT STEM Major', 'STEM Major'])
ax.set_yticklabels(['NOT STEM Major', 'STEM Major'], rotation=90)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Mean Confusion Matrix (LRM Using Only Demographic Features)", fontsize=14)
ax.grid(False)

plt.tight_layout()
plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/LRM_conf_matrix.svg", format='svg')
plt.show()

#%% Simple Logistic Regresion Using only Pre-Pruning Data

#For calulating gain, we can just do (AUC_RFC - AUC_LR)/(AUC_LR)

X = XandY.copy()
X = X.drop(columns=["final_major_t1_STEM", "tier1_STEM_drop_tozero", "ID"]) 

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
# print("Logistic Regression Metric Summary (mean, CI):")
for k, (m, ci) in metric_summary_LR.items():
    print(f"{k:>18}: {m:.3f} ({m-ci:.3f}, {m+ci:.3f})")

print("\nCoefficient / Odds Ratio summary:")
with pd.option_context('display.max_columns', None):
    print(coef_summary)
#%% LRM Plotting Pre-Pruning
### Plot ROC AUC for Logistic Regression ###

# Common FPR grid for interpolation
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []

# Interpolate all TPRs onto mean FPR grid
for fpr, tpr in roc_curves:
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)

interp_tprs = np.array(interp_tprs)
mean_tpr = np.mean(interp_tprs, axis=0)
CI_tpr = conf_int(accuracies)  # Used for plotting 95% CI band width
mean_tpr[-1] = 1.0

# Confidence bounds
tpr_upper = np.minimum(mean_tpr + CI_tpr, 1)
tpr_lower = np.maximum(mean_tpr - CI_tpr, 0)

# Mean AUC and CI
mean_auc = np.mean(auc_scores)
CI_auc = conf_int(auc_scores)
lower_auc = mean_auc - CI_auc
upper_auc = mean_auc + CI_auc

auc_label = f"Mean ROC (AUC = {mean_auc:.3f} [{lower_auc:.3f}, {upper_auc:.3f}])"

# Plot all ROC curves
print(f"Number of individual ROC curves plotted: {len(roc_curves)}")
for fpr, tpr in roc_curves:
    plt.plot(fpr, tpr, alpha=0.1, color='gray')

# Mean ROC
plt.plot(mean_fpr, mean_tpr, color='blue', label=auc_label, linewidth=2)
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=.3, label="95% Confidence Interval")
plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)

plt.title("ROC Curve for Logistic Regression Model", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/LRM_roc_curve_PP.svg", format='svg')
plt.show()


### Plot Mean Confusion Matrix ###

# Create labels for mean ± 95CI
labels = np.empty_like(mean_conf_matrix, dtype=object)
for i in range(2):
    for j in range(2):
        mean_val = mean_conf_matrix[i, j]
        ci_val = conf_matrix_ci[i, j]
        labels[i, j] = f"{mean_val:.1f}\n({mean_val - ci_val:.1f}, {mean_val + ci_val:.1f})"

fig, ax = plt.subplots(figsize=(6, 5))

# Create grid
x = np.arange(3)
y = np.arange(3)

# Plot confusion matrix
c = ax.pcolormesh(x, y, mean_conf_matrix, cmap='Blues', shading='auto')
fig.colorbar(c, ax=ax)

# Add annotations
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.5, labels[i, j], ha='center', va='center', color='black', fontsize=12)

# Axis formatting
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(['NOT STEM Major', 'STEM Major'])
ax.set_yticklabels(['NOT STEM Major', 'STEM Major'], rotation=90)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Mean Confusion Matrix (LRM Using Only Full [Unpruned] Dataset)", fontsize=14)
ax.grid(False)

plt.tight_layout()
plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/LRM_conf_matrix_PP.svg", format='svg')
plt.show()




#%% Pre-Pruning Base Model

X = XandY.copy()
X = X.drop(columns=["final_major_t1_STEM", "tier1_STEM_drop_tozero", "ID"]) 

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
plt.title("ROC Curve for RFC Trained on Full (Unpruned) Dataset", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="lower right", fontsize=8)
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
ax.set_title("Mean Confusion Matrix (RFC using Full [Unpruned] Feature Set)", fontsize=14)

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

importance_df = importance_df.rename(columns={
    'negative_emotionality' : "Negative Emotionality/Neuroticism", 'extraversion': "Extraversion", 'agreeableness': "Agreeableness",
    'conscientiousness': "Conscientiousness", 'open_mindedness': "Open Mindedness", 'gender_F': "Is Female", 'race_MLT': "Is Multiracial",
    'race_CWH': "Is Caucasian/White", 'race_AFB': "Is Afrian American/Black", 'race_AAA': "Is Asian American", 'race_NAT': "Is Native American",
    'race_HPI': "Is Hawaiian/Pacific Islander", 'race_OTH': "Is 'Other' Race", 'ethnicity_HS': "Is Hispanic", 'grade_importance': "SR Importance of Grades",
    'PA_nondenseBaseline_overall': "Baseline nondense Positive Affect", 'NA_nondenseBaseline_overall': "Baseline nondense Negative Affect",
    'studyTotal': "Total Number of People Studied with", 'term_GPA': "GPA Achieved GPA  Achieved during Semester of Study Participation",
    'UM_credits_at_study': "Number of Credits Earned in UM at time of Study Participation", 'GAD_score': "GAD Score", 'PHQ_score': "PHQ Score",
    'SHAPS_score':"SHAPS Score", 'PTQ_score':"PTQ Score", 'EPSI_Tt_score':"EPSI Score", 'SIR_Tt_score':"SIR Score", 'PSWQ_score':"PSWQ Score",
    'SIAS_score':"SIAS Score", 'OCIR_score':"OCIR Score", 'ASRM_score':"ASRM Score", 'PQB_score':"PQB Score", 'MSPSS_Tt_score':"MSPSS Score",
    'CSWS_Tt_score':"CSWS Score", 'semester_study':"Number of Semesters in UM at time of Study", 'Goal_grade_mean':"Mean Goal Grade",
    'Goal_grade_sd':"Standard Deviation of Goal Grade", 'grade_100_mean':"Mean Grade", 'grade_100_sd':"Standard Deviation of Grades",
    'grade_minus_goal_mean':"Mean Grade minus Goal", 'grade_minus_goal_sd':"Standard Deviation of Grade minus Goal",
    'grade_minus_need_mean': "Mean Grade minus Need", 'grade_minus_need_sd':"Standard Deviation of Grade minus Need",
    'NA_dense_mean':"Mean Dense Period Negative Affect", 'NA_dense_sd':"Standard Deviation of Dense Period Negative Affect",
    'need_grade_mean':"Mean Needed Grade", 'need_grade_sd':"Standard Deviation of Needed Grade",
    'negative_mood_mean':"Mean Nondense Period Negative Affect", 'negative_mood_sd':"Standard Deviation of Nondense Period Negative Affect",
    'PA_dense_mean':"Mean Dense Period Positive Affect", 'PA_dense_sd':"Standard Deviation of Dense Period Positive Affect",
    'PE_mean':"Mean Prediction Error", 'PE_sd':"Standard Deviation of Prediction Error",
    'positive_mood_mean':"Mean Nondense Period Positive Affect", 'positive_mood_sd':"Standard Deviation of Nondense Period Positive Affect",
    'pred_100_mean':"Mean Predicted Grade", 'pred_100_sd':"Standard Deviation of Predicted Grade",
    'Prediction_confidence_mean':"Mean Prediction Confidence", 'Prediction_confidence_sd':"Standard Deviation of Prediction Confidence",
    'studyTotal_mean':"Mean Number of People Studied with", 'studyTotal_sd':"Standard Deviation of Number of People Studied with"})

mean_importance = importance_df.mean(skipna=True)
std_importance = importance_df.std(skipna=True)
LCI95_importance = mean_importance.copy()
CI95_importance = mean_importance.copy()
UCI95_importance = mean_importance.copy()
for c in importance_df.columns:
    LCI95_importance[c] = mean_importance[c] - (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
    UCI95_importance[c] = mean_importance[c] + (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
    CI95_importance[c] = (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
    
# Sort by importance
mean_importance = mean_importance.sort_values(ascending=False)
std_importance = std_importance[mean_importance.index]


# Set up figure and axis
fig, ax = plt.subplots(figsize=(18, 9))

# Bar plot with error bars
mean_importance.plot.bar(
    yerr=CI95_importance,
    capsize=4,
    color='steelblue',
    edgecolor='black',
    ax=ax
)
# Labels and title
ax.set_title("Mean Feature Importance of Features from RFC trained on Full (Unpruned) Feature Set", fontsize=14)
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
    plt.plot(fpr, tpr, alpha=.1, color='gray')

# Mean curve
plt.plot(mean_fpr, mean_tpr, color='blue', label=auc_label, linewidth=1)

# 95CI band
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='darkblue', alpha=1, label="95% Confidence Interval")

plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)

# Labels and layout
plt.title("ROC Curve for RFC Trained on Pruned Dataset", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="lower right", fontsize=8)
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
ax.set_title("Mean Confusion Matrix (RFC using Pruned Feature Set)", fontsize=14)

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
LCI95_importance = mean_importance.copy()
CI95_importance = mean_importance.copy()
UCI95_importance = mean_importance.copy()
for c in current_features:
    LCI95_importance[c] = mean_importance[c] - (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
    UCI95_importance[c] = mean_importance[c] + (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
    CI95_importance[c] = (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
    
# Set up figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Bar plot with error bars
mean_importance.plot.bar(
    yerr=CI95_importance,
    capsize=4,
    color='steelblue',
    edgecolor='black',
    ax=ax
)
# Labels and title
ax.set_title("Mean Feature Importance of Pruned Feature Set", fontsize=14)
ax.set_ylabel("Mean Permutation Importance", fontsize=14)
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
#%% Run Base Model (Post-Pruning, Classify DTZ)
#now take the remaining features and run 100 models using only those 10
current_features = ['extraversion', 'open_mindedness', 'term_GPA', 'UM_credits_at_study', 'semester_study', 'Goal_grade_sd', 'grade_100_mean', 'grade_minus_goal_mean', 'NA_dense_mean', 'pred_100_mean']


X = XandY.copy()
X = X[current_features]

y = XandY.copy()
y = XandY["final_major_t1_STEM"]

y2 = XandY.copy()
y2 = XandY["tier1_STEM_drop_tozero"]



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

# Storage
roc_curves2 = []
auc_scores2 = []
accuracies2 = []
precisions2 = []
recalls2 = []
sensitivities2 = []
specificities2 = []
f1_scores2 = []
conf_matrices2 = []
feature_rows2 = []

for seed in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        y2_test = y2.iloc[test_idx]

        feature_names = list(X_train.columns)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        y2_pred = model.predict(X_test)
        y2_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_proba))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        conf_matrices.append(confusion_matrix(y_test, y_pred))
        
        accuracies2.append(accuracy_score(y2_test, y2_pred))
        auc_scores2.append(roc_auc_score(y2_test, y2_proba))
        precisions2.append(precision_score(y2_test, y2_pred))
        recalls2.append(recall_score(y2_test, y2_pred))
        f1_scores2.append(f1_score(y2_test, y2_pred))
        conf_matrices2.append(confusion_matrix(y2_test, y2_pred))
        
        # Sensitivity and Specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivities.append(tp / (tp + fn))
        specificities.append(tn / (tn + fp))
        
        conf_matrices.append(np.array([[tn, fp], [fn, tp]]))
        
        
        tn2, fp2, fn2, tp2 = confusion_matrix(y2_test, y2_pred).ravel()
        sensitivities2.append(tp2 / (tp2 + fn2))
        specificities2.append(tn2 / (tn2 + fp2))
        
        conf_matrices2.append(np.array([[tn2, fp2], [fn2, tp2]]))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append((fpr, tpr))
        
        fpr2, tpr2, _ = roc_curve(y2_test, y2_proba)
        roc_curves2.append((fpr2, tpr2))

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

metric_summary_pRFC2 = {
    "Accuracy": (mean(accuracies2), conf_int(accuracies2)),
    "ROC AUC": (mean(auc_scores2), conf_int(auc_scores2)),
    "Precision": (mean(precisions2), conf_int(precisions2)),
    "Recall (Sensitivity)": (mean(recalls2), conf_int(recalls2)),
    "Specificity": (mean(specificities2), conf_int(specificities2)),
    "F1 Score": (mean(f1_scores2), conf_int(f1_scores2))
}

# Mean confusion matrix
mean_conf_matrix = np.mean(conf_matrices, axis=0)
conf_matrix_ci = conf_int(conf_matrices)

mean_conf_matrix2 = np.mean(conf_matrices2, axis=0)
conf_matrix_ci2 = conf_int(conf_matrices2)
#%% Plot Figures (Post-Pruning, Classify DTZ)
### Plot ROC AUC ###

# Common FPR grid for interpolation
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = []

# Interpolate all TPRs onto mean FPR grid
for fpr, tpr in roc_curves2:
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
mean_auc = np.mean(auc_scores2)

# All ROC curves
print(f"Number of individual ROC curves plotted: {len(roc_curves2)}")

mean_auc = np.mean(auc_scores2)
CI_auc = conf_int(auc_scores2)
lower_auc = mean_auc - CI_auc
upper_auc = mean_auc + CI_auc

auc_label = f"Mean ROC (AUC = {mean_auc:.3f} [{lower_auc:.3f}, {upper_auc:.3f}])"

for fpr, tpr in roc_curves2:
    plt.plot(fpr, tpr, alpha=.1, color='gray')

# Mean curve
plt.plot(mean_fpr, mean_tpr, color='blue', label=auc_label, linewidth=1)

# 95CI band
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='darkblue', alpha=1, label="95% Confidence Interval")

plt.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)

# Labels and layout
plt.title("ROC Curve for RFC Trained on Pruned Dataset (DTZ)", fontsize=14)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/100roc_curve_DTZ.svg", format='svg')  # Save in vector format
plt.show()
### Plot Confusion Matrix + Print Model Performance ###


# Create labels for mean ± 95CI
labels = np.empty_like(mean_conf_matrix2, dtype=object)
for i in range(2):
    for j in range(2):
        mean_val = mean_conf_matrix2[i, j]
        ci_val = conf_matrix_ci2[i, j]
        labels[i, j] = f"{mean_val:.1f}\n({mean_val-ci_val:.1f}, {mean_val+ci_val:.1f})"

fig, ax = plt.subplots(figsize=(6, 5))
# Coordinates for pcolormesh (requires edges, so +1 shape)
x = np.arange(3)
y = np.arange(3)

# Plot confusion matrix as vector-based pcolormesh
c = ax.pcolormesh(x, y, mean_conf_matrix2, cmap='Blues', shading='auto')

# Add colorbar
fig.colorbar(c, ax=ax)

# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.5, labels[i, j], ha='center', va='center', color='black', fontsize=12)

# Axes settings
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(['Did NOT drop STEM Major', 'Did Drop STEM Major'])
ax.set_yticklabels(['Did NOT drop STEM Major', 'Did Drop STEM Major'], rotation=90)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Mean Confusion Matrix (RFC using Pruned Feature Set; DTZ)", fontsize=14)

# Clean look: turn off spines and grid
ax.grid(False)
# for spine in ax.spines.values():
#     spine.set_visible(False)

plt.tight_layout()

# Save as SVG (should have no PNGs)
plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/mean_confusion_matrix_DTZ.svg", format='svg')
plt.show()

print("=== Model Performance Metrics (mean ± 95 CI) ===")
for metric, (m, CI) in metric_summary_pRFC2.items():
    lci = m - CI
    uci = m + CI
    print(f"{metric}: {m:.3f} ({lci:.3f}, {uci:.3f})")
    
### Plot Feature Importances ### Irrelevant, the features are the same
# importance_df = pd.DataFrame(feature_rows)

# mean_importance = importance_df.mean(skipna=True)
# std_importance = importance_df.std(skipna=True)

# # Sort by importance
# mean_importance = mean_importance.sort_values(ascending=False)
# std_importance = std_importance[mean_importance.index]
# LCI95_importance = mean_importance.copy()
# CI95_importance = mean_importance.copy()
# UCI95_importance = mean_importance.copy()
# for c in current_features:
#     LCI95_importance[c] = mean_importance[c] - (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
#     UCI95_importance[c] = mean_importance[c] + (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
#     CI95_importance[c] = (1.96*(std_importance[c]/(importance_df[c].count() ** 0.5)))
    
# # Set up figure and axis
# fig, ax = plt.subplots(figsize=(12, 6))

# # Bar plot with error bars
# mean_importance.plot.bar(
#     yerr=CI95_importance,
#     capsize=4,
#     color='steelblue',
#     edgecolor='black',
#     ax=ax
# )
# # Labels and title
# ax.set_title("Mean Feature Importance of Pruned Feature Set (DTZ)", fontsize=14)
# ax.set_ylabel("Mean Permutation Importance", fontsize=14)
# ax.set_xticklabels(mean_importance.index, rotation=45, ha='right', fontsize=10)
# ax.tick_params(axis='y', labelsize=10)

# # Grid behind bars
# ax.set_axisbelow(True)
# ax.grid(axis='y', linestyle='--', linewidth=0.5)

# # Emphasize y=0 line
# ax.axhline(0, color='black', linewidth=1.2)

# # Layout and save
# plt.tight_layout()
# plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/feature_importance_DTZ.svg", format='svg')
# plt.show()

#%% Calculate and Print Gain

print("\nRelative to the LR model, the Unpruned RFC had improved ROC_AUC by",
      per_gain(metric_summary_ppRFC["ROC AUC"][0], metric_summary_LR["ROC AUC"][0]), "and improved accuracy by",
      per_gain(metric_summary_ppRFC["Accuracy"][0], metric_summary_LR["Accuracy"][0]))

print("\nRelative to the LR model, the Pruned RFC had improved ROC_AUC by",
      per_gain(metric_summary_pRFC["ROC AUC"][0], metric_summary_LR["ROC AUC"][0]), "and improved accuracy by",
      per_gain(metric_summary_pRFC["Accuracy"][0], metric_summary_LR["Accuracy"][0]))

print("\nRelative to the Unpruned RFC, the Pruned RFC had improved ROC_AUC by",
      per_gain(metric_summary_pRFC["ROC AUC"][0], metric_summary_ppRFC["ROC AUC"][0]), "and improved accuracy by",
      per_gain(metric_summary_pRFC["Accuracy"][0], metric_summary_ppRFC["Accuracy"][0]))

#%%

# Compute correlation
XandY2 = XandY.drop(columns=["ID"])
corr = XandY2.corr()

# Manual cluster assignments 
cluster_assignments = {
    #Personality
    'negative_emotionality' : 0, 'extraversion' : 0, 'agreeableness' : 0, 'conscientiousness' : 0, 'open_mindedness' : 0,
    #Demographics
    'gender_F' : 1, 'race_MLT' : 1, 'race_CWH' : 1, 'race_AFB' : 1, 'race_AAA' : 1, 'race_NAT' : 1, 'race_HPI' : 1, 'race_OTH' : 1, 'ethnicity_HS' : 1,
    #Grade Related
    'grade_importance' : 2, 'grade_100_mean' : 2, 'grade_100_sd' : 2, 'grade_minus_goal_mean' : 2, 'grade_minus_goal_sd' : 2, 'grade_minus_need_mean' : 2,
    'grade_minus_need_sd' : 2, 'Goal_grade_mean' : 2, 'Goal_grade_sd' : 2, 'need_grade_mean' : 2, 'need_grade_sd' : 2, 'PE_mean' : 2,
    'PE_sd' : 2, 'pred_100_mean' : 2, 'pred_100_sd' : 2, 'Prediction_confidence_mean' : 2,'Prediction_confidence_sd' : 2,
    #EMA Collected Emotion
    'PA_nondenseBaseline_overall' : 3, 'NA_nondenseBaseline_overall' : 3, 'NA_dense_mean' : 3, 'NA_dense_sd' : 3, 'negative_mood_mean' : 3, 'negative_mood_sd' : 3,
    'PA_dense_mean' : 3, 'PA_dense_sd' : 3, 'positive_mood_mean' : 3, 'positive_mood_sd' : 3,    
    #Exam-Varying Social
    'studyTotal' : 4, 'studyTotal_mean' : 4, 'studyTotal_sd' : 4,
    #Academic Information
    'term_GPA' : 5, 'UM_credits_at_study' : 5, 'semester_study' : 5,
    #Baseline Self-Report Questionnaires
    'GAD_score' : 6, 'PHQ_score' : 6, 'SHAPS_score' : 6, 'PTQ_score' : 6, 'EPSI_Tt_score' : 6, 'SIR_Tt_score' : 6, 'PSWQ_score' : 6, 'SIAS_score' : 6, 'OCIR_score' : 6,
    'ASRM_score' : 6, 'PQB_score' : 6, 'MSPSS_Tt_score' : 6, 'CSWS_Tt_score' : 6,
    #Outcome
    'tier1_STEM_drop_tozero' : 7, 'final_major_t1_STEM' : 7
}

name_map = {
    'negative_emotionality' : "NE/Neuroticism",'extraversion': "Extraversion", 'agreeableness': "Agreeableness",
    'conscientiousness': "Conscientiousness", 'open_mindedness': "Open Mindedness", 'gender_F': "Is Female",
    'race_MLT': "Is Multiracial",'race_CWH': "Is Caucasian/White", 'race_AFB': "Is Afrian American/Black",
    'race_AAA': "Is Asian American", 'race_NAT': "Is Native American", 'race_HPI': "Is Hawaiian/Pacific Islander",
    'race_OTH': "Is 'Other' Race", 'ethnicity_HS': "Is Hispanic", 'grade_importance': "SR Importance of Grades",
    'PA_nondenseBaseline_overall': "Baseline nondense Positive Affect", 'NA_nondenseBaseline_overall': "Baseline nondense Negative Affect",
    'studyTotal': "Total Number of People Studied with", 'term_GPA': "GPA Achieved during Study Semester",
    'UM_credits_at_study': "Number of Credits Earned at Study", 'GAD_score': "GAD Score",
    'PHQ_score': "PHQ Score", 'SHAPS_score':"SHAPS Score",'PTQ_score':"PTQ Score", 'EPSI_Tt_score':"EPSI Score", 'SIR_Tt_score':"SIR Score",
    'PSWQ_score':"PSWQ Score", 'SIAS_score':"SIAS Score", 'OCIR_score':"OCIR Score", 'ASRM_score':"ASRM Score", 'PQB_score':"PQB Score",
    'MSPSS_Tt_score':"MSPSS Score",'CSWS_Tt_score':"CSWS Score", 'semester_study':"Study Semester Number",
    'Goal_grade_mean':"Mean Goal Grade", 'Goal_grade_sd':"SD of Goal Grade",
    'grade_100_mean':"Mean Grade", 'grade_100_sd':"SD of Grades", 'grade_minus_goal_mean':"Mean Grade minus Goal",
    'grade_minus_goal_sd':"SD of Grade minus Goal", 'grade_minus_need_mean': "Mean Grade minus Need",
    'grade_minus_need_sd':"SD of Grade minus Need", 'NA_dense_mean':"Mean Dense Period Negative Affect",
    'NA_dense_sd':"SD of Dense Period Negative Affect", 'need_grade_mean':"Mean Needed Grade",
    'need_grade_sd':"SD of Needed Grade",'negative_mood_mean':"Mean Nondense Period Negative Affect",
    'negative_mood_sd':"SD of Nondense Period Negative Affect", 'PA_dense_mean':"Mean Dense Period Positive Affect",
    'PA_dense_sd':"SD of Dense Period Positive Affect", 'PE_mean':"Mean Prediction Error", 'PE_sd':"SD of Prediction Error",
    'positive_mood_mean':"Mean Nondense Period Positive Affect",'positive_mood_sd':"SD of Nondense Period Positive Affect",
    'pred_100_mean':"Mean Predicted Grade", 'pred_100_sd':"SD of Predicted Grade",
    'Prediction_confidence_mean':"Mean Prediction Confidence", 'Prediction_confidence_sd':"SD of Prediction Confidence",
    'studyTotal_mean':"Mean Number of People Studied with", 'studyTotal_sd':"SD of Number of People Studied with", 'tier1_STEM_drop_tozero' : "Did or Did not Drop STEM Major",
    'final_major_t1_STEM' : "Final Major Was or Was Not STEM"
}

#Assign cluster labels
cluster_label_map = {
    0: "BFI/Personality",
    1: "Demographics",
    2: "Grade Related",
    3: "EMA Collected Emotion",
    4: "Exam-Varying Social",
    5: "Academic Information", 
    6: "Baseline Self-Report Questionnnaires",
    7: "Outcome"
}

#Sort the vairables by cluster number
sorted_vars = sorted(cluster_assignments.keys(), key=lambda x: cluster_assignments[x])
corr_sorted = corr.loc[sorted_vars, sorted_vars]

#Prepare presets for the graph
display_names = [name_map[var] for var in sorted_vars]
display_nums = list(range(1, XandY2.shape[1] + 1))

unique_clusters = sorted(set(cluster_assignments.values()))
palette = sns.color_palette("Set2", len(unique_clusters))
cluster_color_map = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}

row_colors = [cluster_color_map[cluster_assignments[var]] for var in sorted_vars]
col_colors = row_colors  # same order since matrix is symmetric

## Create the clustermap
g = sns.clustermap(
    corr_sorted,
    cmap ="vlag",
    vmin = -1, vmax = 1,                 # Ensures color scale is symmetric
    row_cluster = False,
    col_cluster = False,
    row_colors = row_colors,
    col_colors = row_colors,
    xticklabels = display_nums,
    yticklabels = display_nums,
    figsize=(12, 12)
)

# g.cax.set_visible(False)

# Rotate x labels
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=8)
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=8)

# Create legend for clusters
handles = [
    Patch(facecolor=cluster_color_map[c], label=cluster_label_map[c])
    for c in unique_clusters
]

# Add legend outside the plot
g.ax_heatmap.legend(
    handles=handles,
    title="Variable Cluster",
    loc='lower left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0.
)

# plt.savefig("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/Plots/Pub_figs/correlation_heatmap2.svg", format="svg", bbox_inches="tight", transparent=True)

plt.show()

#%%
#Get the correaltion between the two outcome variables
corrXY = XandY.corr()
print(corrXY.loc["tier1_STEM_drop_tozero","final_major_t1_STEM"])

#%%
#Get the correlation of the specific variables of interset
current_features = ['extraversion', 'open_mindedness', 'term_GPA', 'UM_credits_at_study', 'semester_study', 'Goal_grade_sd', 'grade_100_mean', 'grade_minus_goal_mean', 'NA_dense_mean', 'pred_100_mean']
X = XandY.copy()
X = X[current_features]

corrX = X.corr()
print(corrX) # This is table S3