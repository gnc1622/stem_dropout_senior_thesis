# Pre-load and Check Sample Statistics -----------------------------------------------------------
library(tidyverse)
library(ggplot2)
library(dplyr)
library(rlang)
library(survival)
library(gganimate)
library(transformr)
library(plotly)
library(reshape2)
library(knitr)
library(rmarkdown)
library(patchwork)
library(glmnet)
library(sjPlot)
library(contsurvplot)
library(survMisc)
library(riskRegression)
library(pammtools)
library(survminer)
library(viridis)
library("tibble")
library("ggpubr")
library("patchwork")
library("Matrix")
library("lmerTest")
library("lme4")
library("tibble")
library("RColorBrewer")

#This helps to set the colors for graphs later, is built to be red-green colorblind friendly.
cb_palette <- c(
  "#E69F00",  # orange
  "#56B4E9",  # sky blue
  "#009E73",  # bluish green
  "#F0E442",  # yellow
  "#0072B2",  # dark blue
  "#D55E00",  # vermillion
  "#000000",  # black (sharp contrast)
  "#1E90FF"   # dodger blue (vivid and distinct from dark blue)
)

#Load in survival dataset
sobreviver <- read.csv('//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver.csv', stringsAsFactors = FALSE, na.strings = "NA")
sobreviver_comp <- subset(sobreviver, final_tt_credits >= 120) # >= 120 credits, means able to graduate

#This is the goal grade "deep dive" dataset, will be used later in script
gmx <- c("ID", "Repeat_ID", "Goal_grade_mean", "Goal_grade_sd", "grade_100_mean", "grade_100_sd", "grade_minus_goal_mean", "grade_minus_goal_sd",
         "grade_minus_need_mean", "grade_minus_need_sd", "need_grade_mean", "need_grade_sd", "semester_drop", "tier1_STEM_drop_tozero") 
sobreviver_comp_gmx <- sobreviver_comp[,gmx]

{
  #This will all come from the the XY_reFull_II.csv that gets averaged down into the survival analysis csv
  wide2 <- read.csv('//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_reFull_II.csv', stringsAsFactors = FALSE, na.strings = "NA")
  
  
  #Create new variables in gmx dataframe
  {
    sobreviver_comp_gmx$Goal_grade_exam1 <- c(NA)
    sobreviver_comp_gmx$Goal_grade_exam2 <- c(NA)
    sobreviver_comp_gmx$Goal_grade_exam3 <- c(NA)
    sobreviver_comp_gmx$Goal_grade_exam4 <- c(NA)
    
    sobreviver_comp_gmx$grade_minus_goal_exam1 <- c(NA)
    sobreviver_comp_gmx$grade_minus_goal_exam2 <- c(NA)
    sobreviver_comp_gmx$grade_minus_goal_exam3 <- c(NA)
    sobreviver_comp_gmx$grade_minus_goal_exam4 <- c(NA)
    
    sobreviver_comp_gmx$need_grade_exam1 <- c(NA)
    sobreviver_comp_gmx$need_grade_exam2 <- c(NA)
    sobreviver_comp_gmx$need_grade_exam3 <- c(NA)
    sobreviver_comp_gmx$need_grade_exam4 <- c(NA)
    
    sobreviver_comp_gmx$grade_minus_need_exam1 <- c(NA)
    sobreviver_comp_gmx$grade_minus_need_exam2 <- c(NA)
    sobreviver_comp_gmx$grade_minus_need_exam3 <- c(NA)
    sobreviver_comp_gmx$grade_minus_need_exam4 <- c(NA)
    
    sobreviver_comp_gmx$delta_Goal_grade_exam2 <- c(NA)
    sobreviver_comp_gmx$delta_Goal_grade_exam3 <- c(NA)
    sobreviver_comp_gmx$delta_Goal_grade_exam4 <- c(NA)
    
    sobreviver_comp_gmx$delta_grade_minus_goal_exam2 <- c(NA)
    sobreviver_comp_gmx$delta_grade_minus_goal_exam3 <- c(NA)
    sobreviver_comp_gmx$delta_grade_minus_goal_exam4 <- c(NA)
    
    sobreviver_comp_gmx$delta_need_grade_exam2 <- c(NA)
    sobreviver_comp_gmx$delta_need_grade_exam3 <- c(NA)
    sobreviver_comp_gmx$delta_need_grade_exam4 <- c(NA)
    
    sobreviver_comp_gmx$delta_grade_minus_need_exam2 <- c(NA)
    sobreviver_comp_gmx$delta_grade_minus_need_exam3 <- c(NA)
    sobreviver_comp_gmx$delta_grade_minus_need_exam4 <- c(NA)
  }
  
  #Assign data from refull, should be ready for analysis loop afterwords
  for (i in unique(sobreviver_comp_gmx$ID)) {
    sobreviver_comp_gmx$Goal_grade_exam1[which(sobreviver_comp_gmx$ID == i)] <- wide2$Goal_grade_exam1[which(wide2$ID == i)]
    sobreviver_comp_gmx$Goal_grade_exam2[which(sobreviver_comp_gmx$ID == i)] <- wide2$Goal_grade_exam2[which(wide2$ID == i)]
    sobreviver_comp_gmx$Goal_grade_exam3[which(sobreviver_comp_gmx$ID == i)] <- wide2$Goal_grade_exam3[which(wide2$ID == i)]
    sobreviver_comp_gmx$Goal_grade_exam4[which(sobreviver_comp_gmx$ID == i)] <- wide2$Goal_grade_exam4[which(wide2$ID == i)]
    
    sobreviver_comp_gmx$grade_minus_goal_exam1[which(sobreviver_comp_gmx$ID == i)] <- wide2$grade_minus_goal_exam1[which(wide2$ID == i)]
    sobreviver_comp_gmx$grade_minus_goal_exam2[which(sobreviver_comp_gmx$ID == i)] <- wide2$grade_minus_goal_exam2[which(wide2$ID == i)]
    sobreviver_comp_gmx$grade_minus_goal_exam3[which(sobreviver_comp_gmx$ID == i)] <- wide2$grade_minus_goal_exam3[which(wide2$ID == i)]
    sobreviver_comp_gmx$grade_minus_goal_exam4[which(sobreviver_comp_gmx$ID == i)] <- wide2$grade_minus_goal_exam4[which(wide2$ID == i)]
    
    sobreviver_comp_gmx$need_grade_exam1[which(sobreviver_comp_gmx$ID == i)] <- wide2$need_grade_exam1[which(wide2$ID == i)]
    sobreviver_comp_gmx$need_grade_exam2[which(sobreviver_comp_gmx$ID == i)] <- wide2$need_grade_exam2[which(wide2$ID == i)]
    sobreviver_comp_gmx$need_grade_exam3[which(sobreviver_comp_gmx$ID == i)] <- wide2$need_grade_exam3[which(wide2$ID == i)]
    sobreviver_comp_gmx$need_grade_exam4[which(sobreviver_comp_gmx$ID == i)] <- wide2$need_grade_exam4[which(wide2$ID == i)]
    
    sobreviver_comp_gmx$grade_minus_need_exam1[which(sobreviver_comp_gmx$ID == i)] <- wide2$grade_minus_need_exam1[which(wide2$ID == i)]
    sobreviver_comp_gmx$grade_minus_need_exam2[which(sobreviver_comp_gmx$ID == i)] <- wide2$grade_minus_need_exam2[which(wide2$ID == i)]
    sobreviver_comp_gmx$grade_minus_need_exam3[which(sobreviver_comp_gmx$ID == i)] <- wide2$grade_minus_need_exam3[which(wide2$ID == i)]
    sobreviver_comp_gmx$grade_minus_need_exam4[which(sobreviver_comp_gmx$ID == i)] <- wide2$grade_minus_need_exam4[which(wide2$ID == i)]
    
    sobreviver_comp_gmx$delta_Goal_grade_exam2[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$Goal_grade_exam1[which(wide2$ID == i)],wide2$Goal_grade_exam2[which(wide2$ID == i)]), na.rm = FALSE)
    sobreviver_comp_gmx$delta_Goal_grade_exam3[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$Goal_grade_exam2[which(wide2$ID == i)],wide2$Goal_grade_exam3[which(wide2$ID == i)]), na.rm = FALSE)
    sobreviver_comp_gmx$delta_Goal_grade_exam4[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$Goal_grade_exam3[which(wide2$ID == i)],wide2$Goal_grade_exam4[which(wide2$ID == i)]), na.rm = FALSE)
    
    sobreviver_comp_gmx$delta_grade_minus_goal_exam2[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$grade_minus_goal_exam1[which(wide2$ID == i)],wide2$grade_minus_goal_exam2[which(wide2$ID == i)]), na.rm = FALSE)
    sobreviver_comp_gmx$delta_grade_minus_goal_exam3[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$grade_minus_goal_exam2[which(wide2$ID == i)],wide2$grade_minus_goal_exam3[which(wide2$ID == i)]), na.rm = FALSE)
    sobreviver_comp_gmx$delta_grade_minus_goal_exam4[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$grade_minus_goal_exam3[which(wide2$ID == i)],wide2$grade_minus_goal_exam4[which(wide2$ID == i)]), na.rm = FALSE)
    
    sobreviver_comp_gmx$delta_need_grade_exam2[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$need_grade_exam1[which(wide2$ID == i)],wide2$need_grade_exam2[which(wide2$ID == i)]), na.rm = FALSE)
    sobreviver_comp_gmx$delta_need_grade_exam3[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$need_grade_exam2[which(wide2$ID == i)],wide2$need_grade_exam3[which(wide2$ID == i)]), na.rm = FALSE)
    sobreviver_comp_gmx$delta_need_grade_exam4[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$need_grade_exam3[which(wide2$ID == i)],wide2$need_grade_exam4[which(wide2$ID == i)]), na.rm = FALSE)
    
    sobreviver_comp_gmx$delta_grade_minus_need_exam2[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$grade_minus_need_exam1[which(wide2$ID == i)],wide2$grade_minus_need_exam2[which(wide2$ID == i)]), na.rm = FALSE)
    sobreviver_comp_gmx$delta_grade_minus_need_exam3[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$grade_minus_need_exam2[which(wide2$ID == i)],wide2$grade_minus_need_exam3[which(wide2$ID == i)]), na.rm = FALSE)
    sobreviver_comp_gmx$delta_grade_minus_need_exam4[which(sobreviver_comp_gmx$ID == i)] <-  sum(c((-1)*wide2$grade_minus_need_exam3[which(wide2$ID == i)],wide2$grade_minus_need_exam4[which(wide2$ID == i)]), na.rm = FALSE)
    
  }
} #Load in data and generate dataset

#The following will provide you with demographic information about the FULL dataset
dem_cols <- which(colnames(sobreviver) %in% c("gender_F","race_CWH","race_NAT","race_AAA","race_AFB","race_HPI","race_OTH","race_MLT","ethnicity_HS"))
for (i in dem_cols) {
  col_name <- colnames(sobreviver)[i]
  count <- sum(sobreviver[[i]] == 1, na.rm = TRUE)
  message(paste0(col_name, ": ", count, " (", count/nrow(sobreviver)*100, "%)"))
}
message("total n = ", nrow(sobreviver))

#The following will provide you with demographic information about the >120 Credits dataset
dem_cols <- which(colnames(sobreviver_comp) %in% c("gender_F","race_CWH","race_NAT","race_AAA","race_AFB","race_HPI","race_OTH","race_MLT","ethnicity_HS"))
for (i in dem_cols) {
  col_name <- colnames(sobreviver_comp)[i]
  count <- sum(sobreviver_comp[[i]] == 1, na.rm = TRUE)
  message(paste0(col_name, ": ", count, " (", count/nrow(sobreviver_comp)*100, "%)"))
}
message("total n = ", nrow(sobreviver_comp))

# Survival Analyses (one variable) -------------------------------------------------------

cname <- colnames(sobreviver_comp)

sobreviver_comp_stats <- expand.grid(
  variable = cname
) |>
  mutate(
    hazard_ratio = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    hr_pval = NA_real_,
    hr_power = NA_real_,
  )

#Limit to variables that were collected/would be available during study semester (term)
coi <- c("negative_emotionality","extraversion","agreeableness","conscientiousness","open_mindedness",
         "PA_nondenseBaseline_overall","NA_nondenseBaseline_overall","studyTotal",
         "term_GPA",#"final_UM_credits",
         "GAD_score","PHQ_score","SHAPS_score","PTQ_score", 
         "EPSI_BE_score","EPSI_Rt_score","EPSI_EE_score","EPSI_Pg_score","EPSI_Tt_score",
         "SIR_Aq_score","SIR_DS_score","SIR_Tt_score","PSWQ_score","SIAS_score","OCIR_score","ASRM_score","PQB_score",
         "MSPSS_SO_score","MSPSS_Fa_score","MSPSS_Fr_score","MSPSS_Tt_score","CSWS_Oa_score",	"CSWS_Ap_score",	"CSWS_Co_score",	"CSWS_Ac_score",
         "CSWS_Fa_score",	"CSWS_Vt_score",	"CSWS_Gl_score",	"CSWS_Tt_score",
         "semester_study","UM_credits_at_study",
         # "crd_load_median","delta_GPA_sum","delta_GPA_sd", 
         "Goal_grade_mean","grade_100_mean","grade_minus_goal_mean","grade_minus_need_mean",
         "NA_dense_mean","need_grade_mean","negative_mood_mean","PA_dense_mean","PE_mean", "pred_100_mean", "positive_mood_mean",
         "Prediction_confidence_mean","studyTotal_mean","final_GPA","race_MLT","race_CWH","race_AFB","race_AAA","race_NAT",
         "race_HPI","race_OTH","ethnicity_HS","grade_importance","gender_F")

#Perform Cox PHM on every variable individually
for (ci in coi) {
  
  formula <- as.formula(paste0("Surv(time = sobreviver_comp$semester_drop, event = sobreviver_comp$tier1_STEM_drop_tozero) ~ ", ci))
  cox_model <- coxph(formula, data = sobreviver_comp)
  summary_model <- summary(cox_model)
  sobreviver_comp_stats$hr_power[sobreviver_comp_stats$variable==ci] <- 1 - pchisq(qchisq(1 - 0.05, summary_model$wald["df"]), summary_model$wald["df"], ncp = summary_model$wald["test"])
  sobreviver_comp_stats$hazard_ratio[sobreviver_comp_stats$variable==ci] <- summary_model$coefficients[, "exp(coef)"]
  sobreviver_comp_stats$ci_lower[sobreviver_comp_stats$variable==ci] <- summary_model$conf.int[, "lower .95"]
  sobreviver_comp_stats$ci_upper[sobreviver_comp_stats$variable==ci] <- summary_model$conf.int[, "upper .95"]
  sobreviver_comp_stats$hr_pval[sobreviver_comp_stats$variable==ci] <- summary_model$coefficients[, "Pr(>|z|)"]
  
}

#### Plot the Results ####
#Print Significant Columns
sig_coi <- as.character(sobreviver_comp_stats$variable[which(sobreviver_comp_stats$variable %in% coi & sobreviver_comp_stats$hr_pval < 0.05)])
print(sig_coi)

## Forest Plots ##

#ALL Variables
ggplot(sobreviver_comp_stats[which(sobreviver_comp_stats$variable %in% coi & !is.infinite(sobreviver_comp_stats$ci_upper)),], aes(x = hazard_ratio, y = variable)) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Continuous Variables, Total_Credits > 120",
    subtitle = "Analysis limited to participants with more than 120 Credits"
  ) +
  theme_minimal(base_size = 13)

#Only Significant (p < .05) Variables
ggplot(sobreviver_comp_stats[which(sobreviver_comp_stats$variable %in% coi & sobreviver_comp_stats$hr_pval < 0.05),],
       aes(x = hazard_ratio, y = variable)) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratios for Significant Predictors of STEM Dropout",
    subtitle = "Analysis limited to participants with at least 120 Credits; only variables with p < .05 are displayed"
  )  +
  geom_text(
    aes(label = paste0("HR = ", round(hazard_ratio, 4)," | p = ", formatC(hr_pval, format = "g", digits = 5))),
    vjust = 1.8,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

## Spaghetti x Gradient Plots ##

#This will only plot one variable at a time. So, make sure to change the variable to whichever one you want to visualize
# and then change the name of the variable in the (sub)title

cox_model <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ grade_minus_goal_mean, data = sobreviver_comp,  x = TRUE)
summary_obj <- summary(cox_model)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]

#"Spaghetti" Plot, probability as a function of the variable, grouped by time point
plot_spaghetti <- plot_surv_at_t(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "grade_minus_goal_mean",
                                 data = sobreviver_comp,
                                 model = cox_model,
                                 title = "Survival Probability over Time by the mean difference of grade and goal",
                                 #conf_int = TRUE,
                                 subtitle = paste("Cox Model p =", formatC(p_val, format = "f", digits = 10), "| Cox HR =", sprintf("%.5f", HR)),
                                 t = 1:8,
                                 gg_theme = theme_minimal()) + scale_color_manual(values = cb_palette)

#Survival curve as variable-defined gradient
plot_gradient <- plot_surv_area(time = "semester_drop",
                                status = "tier1_STEM_drop_tozero",
                                variable = "grade_minus_goal_mean",
                                data = sobreviver_comp,
                                model = cox_model, title = "Survivial Probility at across time as a gradient of the mean difference of grade and goal",
                                subtitle = "Kaplan-Meier Estimator Included",
                                gg_theme = theme_minimal(),
                                start_color= "yellow",
                                end_color= "blue",
                                kaplan_meier = TRUE)

#Combined Survival Spaghetti x Gradient Plots
combined_plot <- plot_spaghetti + plot_gradient + plot_layout(ncol = 2)
combined_plot

## Survival Surface Plot ##

#This will only plot one variable at a time. So, make sure to change the variable to whichever one you want to visualize
# and then change the name of the variable in the (sub)title

#Interactive 3D surface
plot_obj <-  plot_surv_3Dsurface(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "grade_minus_goal_mean",
                                 data = sobreviver_comp,
                                 model = cox_model,
                                 interactive=TRUE,
                                 max_t = 10,
                                 zlab = "grade_minus_goal_mean",
                                 xlab = "Time (Semesters)",
                                 ylab = 'Probability of STEM Persistence')

plot_obj <- plot_obj |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of the mean difference of grade and goal", #(Mean Negative Affect during Dense Sampling Period)
      "<br><sub>",
      "Cox Model p = ", formatC(p_val, format = "f", digits = 10),
      " | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    y= 0.9,
    xanchor = "center"
  )
)

plot_obj

#### Include within semester variation of goal grade family of variables ####

#Prep for survival analysis loop and run it
cname <- colnames(sobreviver_comp_gmx)

sobreviver_comp_gmx_stats <- expand.grid(
  variable = cname
) |>
  mutate(
    hazard_ratio = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    hr_pval = NA_real_,
    hr_power = NA_real_,
  )

coi2 <- c("Goal_grade_mean", "Goal_grade_sd", "grade_100_mean", "grade_100_sd", "grade_minus_goal_mean", "grade_minus_goal_sd",
          "grade_minus_need_mean", "grade_minus_need_sd", "need_grade_mean", "need_grade_sd", "Goal_grade_exam1", "Goal_grade_exam2",
          "Goal_grade_exam3", "Goal_grade_exam4", "grade_minus_goal_exam1", "grade_minus_goal_exam2", "grade_minus_goal_exam3",
          "grade_minus_goal_exam4", "need_grade_exam1", "need_grade_exam2", "need_grade_exam3", "need_grade_exam4", "grade_minus_need_exam1",
          "grade_minus_need_exam2", "grade_minus_need_exam3", "grade_minus_need_exam4", "delta_Goal_grade_exam2", "delta_Goal_grade_exam3",      
          "delta_Goal_grade_exam4", "delta_grade_minus_goal_exam2", "delta_grade_minus_goal_exam3", "delta_grade_minus_goal_exam4",
          "delta_need_grade_exam2", "delta_need_grade_exam3", "delta_need_grade_exam4", "delta_grade_minus_need_exam2",
          "delta_grade_minus_need_exam3", "delta_grade_minus_need_exam4")

for (ci in coi2) {
  
  formula <- as.formula(paste0("Surv(time = sobreviver_comp_gmx$semester_drop, event = sobreviver_comp_gmx$tier1_STEM_drop_tozero) ~ ", ci))
  cox_model <- coxph(formula, data = sobreviver_comp_gmx)
  summary_model <- summary(cox_model)
  sobreviver_comp_gmx_stats$hr_power[sobreviver_comp_gmx_stats$variable==ci] <- 1 - pchisq(qchisq(1 - 0.05, summary_model$wald["df"]), summary_model$wald["df"], ncp = summary_model$wald["test"])
  sobreviver_comp_gmx_stats$hazard_ratio[sobreviver_comp_gmx_stats$variable==ci] <- summary_model$coefficients[, "exp(coef)"]
  sobreviver_comp_gmx_stats$ci_lower[sobreviver_comp_gmx_stats$variable==ci] <- summary_model$conf.int[, "lower .95"]
  sobreviver_comp_gmx_stats$ci_upper[sobreviver_comp_gmx_stats$variable==ci] <- summary_model$conf.int[, "upper .95"]
  sobreviver_comp_gmx_stats$hr_pval[sobreviver_comp_gmx_stats$variable==ci] <- summary_model$coefficients[, "Pr(>|z|)"]
  
}

## Plotting ##

## Forest Plot

ggplot(sobreviver_comp_gmx_stats[which(sobreviver_comp_gmx_stats$variable %in% coi2 & !is.infinite(sobreviver_comp_gmx_stats$ci_upper)),], aes(x = hazard_ratio, y = variable)) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() + scale_y_discrete(limits = rev) +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Continuous Variables, Total_Credits > 120",
    subtitle = "Analysis limited to participants with at least 120 Credits"
  ) + #The commented code below just adds HRs, but since these are a mix of significant and insignificant variables, I don't currently have them loaded
  # geom_text(
  #   aes(label = paste0("HR = ", round(hazard_ratio, 4)," | p = ", formatC(hr_pval, format = "g", digits = 5))),
  #   hjust = 1.8,
  #   size = 3.5,
  #   position = position_dodge(width = 0.6)
  # ) +
  theme_minimal(base_size = 13)

ggplot(sobreviver_comp_gmx_stats[which(sobreviver_comp_gmx_stats$variable %in% coi2 & sobreviver_comp_gmx_stats$hr_pval < 0.05),],
       aes(x = hazard_ratio, y = variable)) + scale_y_discrete(limits = rev) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratios for Significant Predictors of STEM Dropout",
    subtitle = "Analysis limited to participants with at least 120 Credits; only variables with p < .05 are displayed"
  )  +
  geom_text(
    aes(label = paste0("HR = ", round(hazard_ratio, 4)," | p = ", formatC(hr_pval, format = "g", digits = 5))),
    vjust = 1.8,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

sig_coi2 <- as.character(sobreviver_comp_gmx_stats$variable[which(sobreviver_comp_gmx_stats$variable %in% coi2 & sobreviver_comp_gmx_stats$hr_pval < 0.05)])

#Print the significant values that appear in the restricted forest plot
print(sig_coi2)

## Survival Surface plot

#This will only plot one variable at a time. So, make sure to change the variable to whichever one you want to visualize
# and then change the name of the variable in the (sub)title

cox_model <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ grade_minus_goal_exam1, data = sobreviver_comp_gmx,  x = TRUE)
summary_obj <- summary(cox_model)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]

plot_obj <-  plot_surv_3Dsurface(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "grade_minus_goal_exam1",
                                 data = sobreviver_comp_gmx,
                                 model = cox_model,
                                 interactive=TRUE,
                                 max_t = 10,
                                 zlab = "grade_minus_goal_exam1",
                                 xlab = "Time (Semesters)",
                                 ylab = 'Probability of STEM Persistence')

plot_obj <- plot_obj |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of the Difference between goal grade and achieved grade at exam 1", #(Mean Negative Affect during Dense Sampling Period)
      "<br><sub>",
      "Cox Model p = ", formatC(p_val, format = "f", digits = 10),
      " | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    #y= 0.9,
    xanchor = "center"
  )
)

plot_obj

# Survival Analyses (Two-Variable) ----------------------------------------

#Start by setting up a pairwise matrix

coi2.5 <- coi2[which(!coi2 %in% coi)]
coi3 <- c(coi, coi2.5)
n <- length(coi3)
empty_matrix <- matrix(NA_real_, nrow = n, ncol = n,
                       dimnames = list(coi3, coi3))

inter_pval_df <- as.data.frame(empty_matrix) #This will store the p_vals
inter_HR_df <- as.data.frame(empty_matrix) #This will store the HRs
inter_LCI_HR_df <- as.data.frame(empty_matrix) #This will store the HRs Lower CI
inter_UCI_HR_df <- as.data.frame(empty_matrix) #This will store the HRs Upper CI
inter_color_df <- as.data.frame(empty_matrix) #This is purely for tresholding pvals
sig_inter_list <- c(NA)

sobreviver_comp_TPAB <- sobreviver_comp
{
  sobreviver_comp_TPAB$Goal_grade_exam1 <- c(NA)
  sobreviver_comp_TPAB$Goal_grade_exam2 <- c(NA)
  sobreviver_comp_TPAB$Goal_grade_exam3 <- c(NA)
  sobreviver_comp_TPAB$Goal_grade_exam4 <- c(NA)
  
  sobreviver_comp_TPAB$grade_minus_goal_exam1 <- c(NA)
  sobreviver_comp_TPAB$grade_minus_goal_exam2 <- c(NA)
  sobreviver_comp_TPAB$grade_minus_goal_exam3 <- c(NA)
  sobreviver_comp_TPAB$grade_minus_goal_exam4 <- c(NA)
  
  sobreviver_comp_TPAB$need_grade_exam1 <- c(NA)
  sobreviver_comp_TPAB$need_grade_exam2 <- c(NA)
  sobreviver_comp_TPAB$need_grade_exam3 <- c(NA)
  sobreviver_comp_TPAB$need_grade_exam4 <- c(NA)
  
  sobreviver_comp_TPAB$grade_minus_need_exam1 <- c(NA)
  sobreviver_comp_TPAB$grade_minus_need_exam2 <- c(NA)
  sobreviver_comp_TPAB$grade_minus_need_exam3 <- c(NA)
  sobreviver_comp_TPAB$grade_minus_need_exam4 <- c(NA)
  
  sobreviver_comp_TPAB$delta_Goal_grade_exam2 <- c(NA)
  sobreviver_comp_TPAB$delta_Goal_grade_exam3 <- c(NA)
  sobreviver_comp_TPAB$delta_Goal_grade_exam4 <- c(NA)
  
  sobreviver_comp_TPAB$delta_grade_minus_goal_exam2 <- c(NA)
  sobreviver_comp_TPAB$delta_grade_minus_goal_exam3 <- c(NA)
  sobreviver_comp_TPAB$delta_grade_minus_goal_exam4 <- c(NA)
  
  sobreviver_comp_TPAB$delta_need_grade_exam2 <- c(NA)
  sobreviver_comp_TPAB$delta_need_grade_exam3 <- c(NA)
  sobreviver_comp_TPAB$delta_need_grade_exam4 <- c(NA)
  
  sobreviver_comp_TPAB$delta_grade_minus_need_exam2 <- c(NA)
  sobreviver_comp_TPAB$delta_grade_minus_need_exam3 <- c(NA)
  sobreviver_comp_TPAB$delta_grade_minus_need_exam4 <- c(NA)
}
for (i in unique(sobreviver_comp_TPAB$ID)) {
  sobreviver_comp_TPAB$Goal_grade_exam1[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$Goal_grade_exam1[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$Goal_grade_exam2[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$Goal_grade_exam2[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$Goal_grade_exam3[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$Goal_grade_exam3[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$Goal_grade_exam4[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$Goal_grade_exam4[which(sobreviver_comp_gmx$ID == i)]
  
  sobreviver_comp_TPAB$grade_minus_goal_exam1[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$grade_minus_goal_exam1[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$grade_minus_goal_exam2[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$grade_minus_goal_exam2[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$grade_minus_goal_exam3[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$grade_minus_goal_exam3[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$grade_minus_goal_exam4[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$grade_minus_goal_exam4[which(sobreviver_comp_gmx$ID == i)]
  
  sobreviver_comp_TPAB$need_grade_exam1[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$need_grade_exam1[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$need_grade_exam2[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$need_grade_exam2[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$need_grade_exam3[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$need_grade_exam3[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$need_grade_exam4[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$need_grade_exam4[which(sobreviver_comp_gmx$ID == i)]
  
  sobreviver_comp_TPAB$grade_minus_need_exam1[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$grade_minus_need_exam1[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$grade_minus_need_exam2[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$grade_minus_need_exam2[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$grade_minus_need_exam3[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$grade_minus_need_exam3[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$grade_minus_need_exam4[which(sobreviver_comp_TPAB$ID == i)] <- sobreviver_comp_gmx$grade_minus_need_exam4[which(sobreviver_comp_gmx$ID == i)]
  
  sobreviver_comp_TPAB$delta_Goal_grade_exam2[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_Goal_grade_exam2[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$delta_Goal_grade_exam3[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_Goal_grade_exam3[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$delta_Goal_grade_exam4[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_Goal_grade_exam4[which(sobreviver_comp_gmx$ID == i)]
  
  sobreviver_comp_TPAB$delta_grade_minus_goal_exam2[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_grade_minus_goal_exam2[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$delta_grade_minus_goal_exam3[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_grade_minus_goal_exam3[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$delta_grade_minus_goal_exam4[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_grade_minus_goal_exam4[which(sobreviver_comp_gmx$ID == i)]
  
  sobreviver_comp_TPAB$delta_need_grade_exam2[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_need_grade_exam2[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$delta_need_grade_exam3[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_need_grade_exam3[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$delta_need_grade_exam4[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_need_grade_exam4[which(sobreviver_comp_gmx$ID == i)]
  
  sobreviver_comp_TPAB$delta_grade_minus_need_exam2[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_grade_minus_need_exam2[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$delta_grade_minus_need_exam3[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_grade_minus_need_exam3[which(sobreviver_comp_gmx$ID == i)]
  sobreviver_comp_TPAB$delta_grade_minus_need_exam4[which(sobreviver_comp_TPAB$ID == i)] <-  sobreviver_comp_gmx$delta_grade_minus_need_exam4[which(sobreviver_comp_gmx$ID == i)]
}

for (i in coi3) {
  if (!which(coi3 == i)+1 >= length(coi3)) {
    for (j in coi3[(which(coi3 == i)+1):length(coi3)]) {
      #This first part will run the model, we should be maximizing efficiency by dealing with the commutative property... 
      #you should only get one triangle, always i*j {j | j >= i+1}
      formula <- as.formula(paste0("Surv(time = sobreviver_comp_TPAB$semester_drop, event = sobreviver_comp_TPAB$tier1_STEM_drop_tozero) ~ ", i, " * ", j))
      cox_model <- coxph(formula, data = sobreviver_comp_TPAB)
      summary_model <- summary(cox_model)
      
      inter_pval_df[i,j] <- summary_model$coefficients[paste0(i,":",j), "Pr(>|z|)"]
      inter_HR_df[i,j] <- summary_model$coefficients[paste0(i,":",j), "exp(coef)"]
      inter_LCI_HR_df[i,j] <- summary_model$conf.int[paste0(i,":",j), "lower .95"]
      inter_UCI_HR_df[i,j] <- summary_model$conf.int[paste0(i,":",j), "upper .95"]
      
      
      if (is.na(inter_pval_df[i,j])) {
        inter_color_df[i,j] <- NA
      } else if (inter_pval_df[i,j] < 0.001) {
        inter_color_df[i,j] <- 3
      } else if (inter_pval_df[i,j] < 0.01) {
        inter_color_df[i,j] <- 2
      } else if (inter_pval_df[i,j] < 0.05) {
        inter_color_df[i,j] <- 1
        sig_inter_list <- c(sig_inter_list, paste0(i,":",j))
      } else {
        inter_color_df[i,j] <- 0
      }
    }
  }
}
sig_inter_list <- sig_inter_list[which(!is.na(sig_inter_list))]

# Convert p-value matrix to long format
plot_df <- inter_pval_df |>
  rownames_to_column("var1") |>
  pivot_longer(-var1, names_to = "var2", values_to = "pval") |>
  filter(!is.na(pval)) |>
  mutate(
    sig_cat = case_when(
      pval < 0.001 ~ "p < 0.001",
      pval < 0.01  ~ "p < 0.01",
      pval < 0.05  ~ "p < 0.05", 
      TRUE         ~ "ns"
    )
  )

# Prep the other matrices in the same way
cox_hr_df <- inter_HR_df |>
  rownames_to_column("var1") |>
  pivot_longer(-var1, names_to = "var2", values_to = "cox_hr")

uci_df <- inter_UCI_HR_df |>
  rownames_to_column("var1") |>
  pivot_longer(-var1, names_to = "var2", values_to = "UCI")

lci_df <- inter_LCI_HR_df |>
  rownames_to_column("var1") |>
  pivot_longer(-var1, names_to = "var2", values_to = "LCI")

# Join them all together by var1 + var2
plot_df <- plot_df |>
  left_join(cox_hr_df, by = c("var1", "var2")) |>
  left_join(uci_df,     by = c("var1", "var2")) |>
  left_join(lci_df,     by = c("var1", "var2"))

plot_df$var1.f <- factor(plot_df$var1, levels = rev(coi3))
plot_df$var2.f <- factor(plot_df$var2, levels = coi3)

ggplot(plot_df, aes(x = var2, y = var1, fill = sig_cat)) +
  geom_tile(color = "white", width = 1, height = 1) +
  scale_fill_manual(
    values = c("p < 0.001" = cb_palette[3], "p < 0.01" = cb_palette[2], "p < 0.05" = cb_palette[1], "ns" = "gray90"),
    na.value = NA  # ensures NA values don't appear
  ) +
  scale_y_discrete(limits = coi3, expand = c(0, 0)) +  # Flip y-axis for matrix-style orientation
  scale_x_discrete(limits = rev(coi3), expand = c(0, 0)) +       # Ensure axis order matches
  #coord_fixed() +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text(size = 6),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Interaction Significance Matrix, Grade Minus Goal/Need Analysis",
    x = NULL,
    y = NULL,
    fill = "p-value"
  )

#### Interaction Term Cox Forests ####

n <- length(sig_inter_list)
clmns <- c("p_val", "Cox_HR", "CHR_UCI", "CHR_LCI")
m <- length(clmns)
empty_matrix2 <- matrix(NA_real_, nrow = n, ncol = m,
                        dimnames = list(sig_inter_list, clmns))

sig_inter_df <- as.data.frame(empty_matrix2) 

lookup_table <- list(
  p_val   = inter_pval_df,
  Cox_HR  = inter_HR_df,
  CHR_UCI = inter_UCI_HR_df,
  CHR_LCI = inter_LCI_HR_df
)

for (i in seq_len(nrow(sig_inter_df))) {
  inter_names <- strsplit(rownames(sig_inter_df)[i], ":")[[1]]
  inter_milan <- inter_names[1]
  inter_miami <- inter_names[2]
  
  for (j in clmns) {
    sig_inter_df[rownames(sig_inter_df)[i], j] <- lookup_table[[j]][inter_milan, inter_miami]
  }
}

sig_plot_df <- plot_df[which(plot_df$pval < 0.05),]
sig_plot_df$inter_var <- paste0(sig_plot_df$var1,":",sig_plot_df$var2)

sig_plot_HR_01 <- sig_plot_df[which(sig_plot_df$cox_hr >= 1.01 | sig_plot_df$cox_hr <= 0.99),]

#Before I was doing it so that I would get 4 forest plots, but now I think there's too many variables, so instead we'll do 5-8 forest plots

eigth1_n <- floor(nrow(sig_plot_HR_01)/ 8)
eigth2_n <- floor(2*nrow(sig_plot_HR_01)/ 8)
#fifth1_n <- floor(nrow(sig_plot_HR_01)/ 5)
#fourth1_n <- floor(nrow(sig_plot_HR_01)/ 4)
# third1_n <- floor(nrow(sig_plot_HR_01)/3)
eigth3_n <- floor(3*nrow(sig_plot_HR_01)/ 8)
# fifth2_n <- floor(2*nrow(sig_plot_HR_01)/ 5)
eigth4_n <- floor(4*nrow(sig_plot_HR_01)/ 8)
#half_n <- floor(nrow(sig_plot_HR_01)/ 2) #I know eigth4_n and half_n are the same, its just for simplicity's sake, don't worry too much.
# fifth3_n <- floor(3*nrow(sig_plot_HR_01)/ 5)
eigth5_n <- floor(5*nrow(sig_plot_HR_01)/ 8)
# third2_n <- floor(2*nrow(sig_plot_HR_01)/3)
#fourth3_n <- floor(3*nrow(sig_plot_HR_01)/ 4)
eigth6_n <- floor(6*nrow(sig_plot_HR_01)/ 8)
# fifth4_n <- floor(4*nrow(sig_plot_HR_01)/ 5)
eigth7_n <- floor(7*nrow(sig_plot_HR_01)/ 8)
full_n <- nrow(sig_plot_HR_01)

ggplot(sig_plot_HR_01[1:eigth1_n,], aes(x = cox_hr, y = sig_plot_HR_01$inter_var[1:eigth1_n])) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = UCI, xmax = LCI),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Interaction Terms, Continuous Variables, Total_Credits >= 120, p < 0.05 (1/8)",
    subtitle = "Interactions included if they modify risk by at least 1%"
  )  +
  geom_text(
    aes(
      x = UCI,
      label = paste0("HR = ", round(cox_hr, 4), " | p = ", formatC(pval, format = "f", digits = 5))
    ),
    # hjust = -0.2,
    vjust = 1.5,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

ggplot(sig_plot_HR_01[eigth1_n:eigth2_n,], aes(x = cox_hr, y = sig_plot_HR_01$inter_var[eigth1_n:eigth2_n])) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = UCI, xmax = LCI),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Interaction Terms, Continuous Variables, Total_Credits >= 120, p < 0.05 (2/8)",
    subtitle = "Interactions included if they modify risk by at least 1%"
  )  +
  geom_text(
    aes(
      x = UCI,
      label = paste0("HR = ", round(cox_hr, 4), " | p = ", formatC(pval, format = "f", digits = 5))
    ),
    # hjust = -0.2,
    vjust = 1.5,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

ggplot(sig_plot_HR_01[eigth2_n:eigth3_n,], aes(x = cox_hr, y = sig_plot_HR_01$inter_var[eigth2_n:eigth3_n])) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = UCI, xmax = LCI),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Interaction Terms, Continuous Variables, Total_Credits >= 120, p < 0.05 (3/8)",
    subtitle = "Interactions included if they modify risk by at least 1%"
  )  +
  geom_text(
    aes(
      x = UCI,
      label = paste0("HR = ", round(cox_hr, 4), " | p = ", formatC(pval, format = "f", digits = 5))
    ),
    # hjust = -0.2,
    vjust = 1.5,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

ggplot(sig_plot_HR_01[eigth3_n:eigth4_n,], aes(x = cox_hr, y = sig_plot_HR_01$inter_var[eigth3_n:eigth4_n])) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = UCI, xmax = LCI),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Interaction Terms, Continuous Variables, Total_Credits >= 120, p < 0.05 (4/8)",
    subtitle = "Interactions included if they modify risk by at least 1%"
  )  +
  geom_text(
    aes(
      x = UCI,
      label = paste0("HR = ", round(cox_hr, 4), " | p = ", formatC(pval, format = "f", digits = 5))
    ),
    # hjust = -0.2,
    vjust = 1.5,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

ggplot(sig_plot_HR_01[eigth4_n:eigth5_n,], aes(x = cox_hr, y = sig_plot_HR_01$inter_var[eigth4_n:eigth5_n])) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = UCI, xmax = LCI),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Interaction Terms, Continuous Variables, Total_Credits >= 120, p < 0.05 (5/8)",
    subtitle = "Interactions included if they modify risk by at least 1%"
  )  +
  geom_text(
    aes(
      x = UCI,
      label = paste0("HR = ", round(cox_hr, 4), " | p = ", formatC(pval, format = "f", digits = 5))
    ),
    # hjust = -0.2,
    vjust = 1.5,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

ggplot(sig_plot_HR_01[eigth5_n:eigth6_n,], aes(x = cox_hr, y = sig_plot_HR_01$inter_var[eigth5_n:eigth6_n])) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = UCI, xmax = LCI),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Interaction Terms, Continuous Variables, Total_Credits >= 120, p < 0.05 (6/8)",
    subtitle = "Interactions included if they modify risk by at least 1%"
  )  +
  geom_text(
    aes(
      x = UCI,
      label = paste0("HR = ", round(cox_hr, 4), " | p = ", formatC(pval, format = "f", digits = 5))
    ),
    # hjust = -0.2,
    vjust = 1.5,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

ggplot(sig_plot_HR_01[eigth6_n:eigth7_n,], aes(x = cox_hr, y = sig_plot_HR_01$inter_var[eigth6_n:eigth7_n])) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = UCI, xmax = LCI),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Interaction Terms, Continuous Variables, Total_Credits >= 120, p < 0.05 (7/8)",
    subtitle = "Interactions included if they modify risk by at least 1%"
  )  +
  geom_text(
    aes(
      x = UCI,
      label = paste0("HR = ", round(cox_hr, 4), " | p = ", formatC(pval, format = "f", digits = 5))
    ),
    # hjust = -0.2,
    vjust = 1.5,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)

ggplot(sig_plot_HR_01[eigth7_n:full_n,], aes(x = cox_hr, y = sig_plot_HR_01$inter_var[eigth7_n:full_n])) +
  geom_point(position = position_dodge(width = 0.6), size = 3) + scale_x_log10() +
  geom_errorbarh(aes(xmin = UCI, xmax = LCI),
                 height = 0.2,
                 position = position_dodge(width = 0.6)) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(
    x = "Hazard Ratio (HR)",
    y = NULL,
    title = "Hazard Ratio, Interaction Terms, Continuous Variables, Total_Credits >= 120, p < 0.05 (8/8)",
    subtitle = "Interactions included if they modify risk by at least 1%"
  )  +
  geom_text(
    aes(
      x = UCI,
      label = paste0("HR = ", round(cox_hr, 4), " | p = ", formatC(pval, format = "f", digits = 5))
    ),
    # hjust = -0.2,
    vjust = 1.5,
    size = 3.5,
    position = position_dodge(width = 0.6)
  ) +
  theme_minimal(base_size = 13)
#All 8 plots, if you change the number, have fun

# Summative Modeling ------------------------------------------

#Here's a list of variables to select from:
{
  # "term_GPA"                     "final_GPA"
  # "delta_GPA_sum"                "delta_GPA_median"            
  # "delta_GPA_sd"                 "Current_average_mean"        
  # "Current_average_sd"           "Goal_grade_mean"             
  # "Goal_grade_sd"                "grade_100_mean"              
  # "grade_100_sd"                 "grade_minus_goal_mean"       
  # "grade_minus_goal_sd"          "grade_minus_need_mean"       
  # "grade_minus_need_sd"          "need_grade_mean"             
  # "need_grade_sd"                               
  # "PA_dense_sd"                  "PE_mean"                     
  # "PE_sd"                        "pred_100_mean"               
  # "pred_100_sd"                  "Prediction_confidence_mean"  
  # "Prediction_confidence_sd"     
}

cox_model <- coxph(Surv(time = sobreviver_comp$semester_drop, event = sobreviver_comp$tier1_STEM_drop_tozero) ~ grade_minus_goal_mean +
                     grade_minus_goal_sd + Goal_grade_sd + Goal_grade_mean + PE_mean + PE_sd + delta_GPA_sum + delta_GPA_sd +
                     pred_100_sd + Prediction_confidence_mean + Prediction_confidence_sd + term_GPA + final_GPA,
                   data = sobreviver_comp_TPAB)

cox_model <- coxph(Surv(time = sobreviver_comp$semester_drop, event = sobreviver_comp$tier1_STEM_drop_tozero) ~ grade_minus_goal_mean +
                     PE_mean + grade_100_mean, data = sobreviver_comp_TPAB)

summary_model <- summary(cox_model)
summary_model

#grade_minus_need_mean, grade_100_mean (alone) both "remove" significance of grade_minus_goal_mean. Likely due to covariability.
#pred_100_mean also kills signal, but only when added above the others.

# Alphas Survival Analysis ------------------------------------------------

#### Pre-Processing ####

## Individual Learning Rates Set-Up ##

#Make sure to have run this AFTER calculating the alphas in the "Pubq_code.py" script
sobreviver_alpha <- read.csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver_alpha.csv", stringsAsFactors = FALSE, na.strings = "NA")

#I think the line that filters out blank values for alpha_PE may remove those with < 120 credits implicitly, but I just want to be absolutely certain
sobreviver_alpha <- sobreviver_alpha[which(sobreviver_alpha$final_tt_credits >= 120),] 
sobreviver_alpha <- sobreviver_alpha[which(!is.na(sobreviver_alpha$alpha_PE)),]

#The following is for the purpose of running an LM later
sobreviver_alpha$final_major_t1_STEM.f <- as.factor(sobreviver_alpha$final_major_t1_STEM)
sobreviver_alpha$tier1_STEM_drop_tozero.f <- as.factor(sobreviver_alpha$tier1_STEM_drop_tozero)

## Global Learning Rates Set-Up ##

#This will look a lot like the previous set-up section so I may omit some of the notes that were written since they are the same as those above
sobreviver_alpha_II <- read.csv("//datastore01.psy.miami.edu/Groups/AHeller_Lab/Undergrad/ANavarro/ST/XY_sobreviver_II_alpha.csv", stringsAsFactors = FALSE, na.strings = "NA")
sobreviver_alpha_II <- sobreviver_alpha_II[which(sobreviver_alpha_II$final_tt_credits >= 120),] 
sobreviver_alpha_II <- sobreviver_alpha_II[which(!is.na(sobreviver_alpha_II$alpha_PE)),]

# sobreviver_alpha_II$final_major_t1_STEM.f <- as.factor(sobreviver_alpha_II$final_major_t1_STEM)
# sobreviver_alpha_II$tier1_STEM_drop_tozero.f <- as.factor(sobreviver_alpha_II$tier1_STEM_drop_tozero)

sobreviver_alpha_II$PE_mc_sign <- c(NA) #this would say whether the person had more negative or positive PEs; we'll be using this to assign alpha_(+/-)
sobreviver_alpha$alpha_sign <- c(NA)
sobreviver_alpha$alpha_sign.b <- c(NA)

for (i in sobreviver_alpha_II$ID) {
  PEs <- c(NA, NA, NA) #only three because there is only three real points of relevant updating: 1->2, 2->3, and 3->4
  PE_signs <- c(NA, NA, NA)
  subdf <- sobreviver_alpha_II[sobreviver_alpha_II$ID == i,]
  PEs <- c(subdf$PE_exam1, subdf$PE_exam2, subdf$PE_exam3)
  PE_signs[which(PEs > 0)] <- 1
  PE_signs[which(PEs < 0)] <- 0
  #If mean > 0.5, then more are positive
  #If mean < 0.5, then more are negative
  #If mean = 0.5, then equally + and - #impossible in this case - update it's not impossible, PE = 0 IS possible leaving only two signed PEs
  if (mean(PE_signs, na.rm = TRUE) > 0.5) {
    sobreviver_alpha_II$PE_mc_sign[sobreviver_alpha_II$ID == i] <- 1
  } else if (mean(PE_signs, na.rm = TRUE) == 0.5) {
    sobreviver_alpha_II$PE_mc_sign[sobreviver_alpha_II$ID == i] <- 0
  } else if (mean(PE_signs, na.rm = TRUE) < 0.5) {
    sobreviver_alpha_II$PE_mc_sign[sobreviver_alpha_II$ID == i] <- -1
  }
  
  if (sobreviver_alpha_II$PE_mc_sign[sobreviver_alpha_II$ID == i] == 1) {
    sobreviver_alpha$alpha_sign[sobreviver_alpha$ID == i] <- -0.1961 #This is the alpha_pos from the Pubq_code.py script (8/25/25)
    sobreviver_alpha$alpha_sign.b[sobreviver_alpha$ID == i] <- 1
  } else if (sobreviver_alpha_II$PE_mc_sign[sobreviver_alpha_II$ID == i] == -1) {
    sobreviver_alpha$alpha_sign[sobreviver_alpha$ID == i] <- 0.0232 #This is the alpha_neg from the Pubq_code.py script (8/25/25)
    sobreviver_alpha$alpha_sign.b[sobreviver_alpha$ID == i] <- 0
  } #We'll leave the 0 case as NA, don't want to mess up the model
}

#This will just tell you sample size for the global learning rates
message("Original n = ", nrow(sobreviver_alpha), " | After removing cases where there are an equal number of + and - PEs, n = ", nrow(sobreviver_alpha[which(!is.na(sobreviver_alpha$alpha_sign)),]),
        " | Therefore, we've only lost: ", nrow(sobreviver_alpha) - nrow(sobreviver_alpha[which(!is.na(sobreviver_alpha$alpha_sign)),]), " particiapnts.")

#### Individual Learning Rates ####

## Linear Model ##

glimpse_PE <- glm(final_major_t1_STEM.f ~ alpha_PE, data = sobreviver_alpha, family = binomial(link = "logit"))
summary(glimpse_PE)
plot_model(glimpse_PE, type = "pred", terms = c("alpha_PE"))

glimpse_GMG <- glm(final_major_t1_STEM.f ~ alpha_GMG, data = sobreviver_alpha, family = binomial(link = "logit"))
summary(glimpse_GMG)
plot_model(glimpse_GMG, type = "pred", terms = c("alpha_GMG"))

## Survival ##

## Alpha Based on PE
cox_model_aPE <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ alpha_PE, data = sobreviver_alpha,  x = TRUE)
summary(cox_model_aPE)
summary_obj <- summary(cox_model_aPE)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]

#Interactive 3D surface
plot_obj <-  plot_surv_3Dsurface(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "alpha_PE",
                                 data = sobreviver_alpha,
                                 model = cox_model_aPE, #title = "Survival Probability x t x PA_dense_mean",
                                 interactive=TRUE,#, subtitle = paste("Cox Model p =", format.pval(p_val, digits = 3))
                                 max_t = 10,
                                 zlab = "alpha_PE",
                                 xlab = "Time (Semesters)",
                                 ylab = 'Probability of STEM Persistence')

plot_obj <- plot_obj |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of alpha_PE", #(Mean Negative Affect during Dense Sampling Period)
      "<br><sub>",
      "Cox Model p = ", formatC(p_val, format = "f", digits = 10),
      " | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    y= 0.9,
    xanchor = "center"
  )
)

plot_obj

## Alpha Based on GMG
cox_model_GMG <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ alpha_GMG, data = sobreviver_alpha,  x = TRUE)
summary(cox_model_GMG)
summary_obj <- summary(cox_model_GMG)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]

#Interactive 3D surface
plot_obj <-  plot_surv_3Dsurface(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "alpha_GMG",
                                 data = sobreviver_alpha,
                                 model = cox_model_GMG, #title = "Survival Probability x t x PA_dense_mean",
                                 interactive=TRUE,#, subtitle = paste("Cox Model p =", format.pval(p_val, digits = 3))
                                 max_t = 10,
                                 zlab = "alpha_GMG",
                                 xlab = "Time (Semesters)",
                                 ylab = 'Probability of STEM Persistence')

plot_obj <- plot_obj |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of alpha_GMG", #(Mean Negative Affect during Dense Sampling Period)
      "<br><sub>",
      "Cox Model p = ", formatC(p_val, format = "f", digits = 10),
      " | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    y= 0.9,
    xanchor = "center"
  )
)

plot_obj

#### Global Learning Rates ####

cox_model_aPE <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ alpha_sign.b, data = sobreviver_alpha,  x = TRUE)
summary_obj <- summary(cox_model_aPE)
summary(cox_model_aPE)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]

cname <- "alpha_sign.b"

#Collect the relevant stats
sobreviver_stats <- expand.grid(
  variable = cname,
  level = 1
) |>
  mutate(
    hazard_ratio = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    hr_pval = NA_real_,
    hr_power = NA_real_,
    km_pval = NA_real_,
    km_power = NA_real_,
    n_events = NA_real_,
    n_sample = NA_real_
  )

sobreviver_stats$hazard_ratio <- summary(cox_model_aPE)$coefficients[1, "exp(coef)"]
sobreviver_stats$ci_lower <- summary(cox_model_aPE)$conf.int[1, "lower .95"]
sobreviver_stats$ci_upper <- summary(cox_model_aPE)$conf.int[1, "upper .95"]
sobreviver_stats$hr_pval <- summary(cox_model_aPE)$coefficients[1, "Pr(>|z|)"]
sobreviver_stats$hr_power <- 1 - pchisq(qchisq(1 - 0.05, summary_obj$wald["df"]), summary_obj$wald["df"], ncp = summary_obj$wald["test"])
sobreviver_stats$n_events <- sum(sobreviver_alpha$tier1_STEM_drop_tozero[which(sobreviver_alpha$alpha_sign.b ==1)], na.rm = TRUE)
sobreviver_stats$n_sample <- length(sobreviver_alpha$tier1_STEM_drop_tozero[which(sobreviver_alpha$alpha_sign.b ==1)])
#You could plot a single hazard ratio on a single tree forest plot, but there's no need, a simple report of the HR is sufficient

# Kaplan-Meier fit & log-rank test
km_fit <- survfit(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ alpha_sign.b, data = sobreviver_alpha)
survdiff_test <- survdiff(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ alpha_sign.b, data = sobreviver_alpha)
chisq <- survdiff_test$chisq
df <- length(survdiff_test$n) - 1
sobreviver_stats$km_pval <- 1 - pchisq(chisq, df)
sobreviver_stats$km_power <- 1 - pchisq(qchisq(1 - 0.05, df), df, ncp = chisq)

#Print Tables
print(sobreviver_stats)
summary(km_fit)$table

#See if median exists (probably won't)
meds <- summary(km_fit)$table[, "median"]
has_median <- any(!is.na(meds))

#Plot
ggsurvplot(km_fit,
           pval = TRUE,
           conf.int = TRUE,
           linetype = "strata",
           surv.median.line = if (has_median) "hv" else NULL,
           palette = c("#E7B800", "#2E9FDF"),
           ggtheme = theme_bw(),
           xlab = "Time (Semesters)", legend = "bottom", legend.title = "Alpha_Sign", legend.labs = c("alpha_neg (0) = 0.0232", "alpha_pos (1) = -0.1961"),
           risk.table = TRUE, risk.table.y.text.col = TRUE, risk.table.col = "strata"#,
           # xlim = c(0,10)
)

















