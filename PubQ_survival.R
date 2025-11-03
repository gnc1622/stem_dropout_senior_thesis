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
library(tibble)
library(ggpubr)
library(patchwork)
library(Matrix)
library(lmerTest)
library(lme4)
library(tibble)
library(RColorBrewer)
library(emmeans)

#Define z_score function
z_score <- function(x, na_rm = TRUE) {
  return((x - mean(x, na.rm = na_rm))/sd(x, na.rm = na_rm))
}

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


# Compute and Plot Survival for Feature Pruning Data ----------------------

#Create Subset of Variables that were identified through Feature Pruning
#We do this subset for two reasons:
##1) n_cross_correction = 10 instead of 80-90+
##2) because the main purpose of this analyses is to understand how the variables identified by the ML as important are behaving over time
MVF <- c('extraversion', 'open_mindedness', 'term_GPA', 'UM_credits_at_study', 'semester_study', 'Goal_grade_sd', 'grade_100_mean', 'grade_minus_goal_mean', 'NA_dense_mean', 'pred_100_mean')

#Set up and compute a z-scored version of the data, this will allow you to cross compare the hazard ratios
sobreviver_comp_z <- sobreviver_comp[MVF]
for (i in 1:length(MVF)) {
  sobreviver_comp_z[[i]] <- z_score(sobreviver_comp_z[[i]])
}
#Add back in the time and outcome variables
sobreviver_comp_z$tier1_STEM_drop_tozero <- sobreviver_comp$tier1_STEM_drop_tozero
sobreviver_comp_z$semester_drop <- sobreviver_comp$semester_drop

#Create DF to store the p values, adjusted p values, and HR; then compute the HR and the UNadjusted p-val
mini_comp_stats <- expand.grid(
  variable = MVF
) |>
  mutate(
    hazard_ratio = NA_real_,
    ci_lower = NA_real_,
    ci_upper = NA_real_,
    hr_pval = NA_real_,
    hr_q_BY = NA_real_,
    #hr_q_BH = NA_real_,
    hr_q_Bonferroni = NA_real_,
    hr_q_FDR = NA_real_,
    hr_q_holm = NA_real_
  )

for (ci in MVF) {
  
  formula <- as.formula(paste0("Surv(time = sobreviver_comp_z$semester_drop, event = sobreviver_comp_z$tier1_STEM_drop_tozero) ~ ", ci))
  cox_model <- coxph(formula, data = sobreviver_comp_z)
  summary_model <- summary(cox_model)
  #mini_comp_stats$hr_power[mini_comp_stats$variable==ci] <- 1 - pchisq(qchisq(1 - 0.05, summary_model$wald["df"]), summary_model$wald["df"], ncp = summary_model$wald["test"])
  mini_comp_stats$hazard_ratio[mini_comp_stats$variable==ci] <- summary_model$coefficients[, "exp(coef)"]
  mini_comp_stats$ci_lower[mini_comp_stats$variable==ci] <- summary_model$conf.int[, "lower .95"]
  mini_comp_stats$ci_upper[mini_comp_stats$variable==ci] <- summary_model$conf.int[, "upper .95"]
  mini_comp_stats$hr_pval[mini_comp_stats$variable==ci] <- summary_model$coefficients[, "Pr(>|z|)"]
  
}

#Compute the adjusted p-vals, the "q-vals"
mini_comp_stats$hr_q_BY <- p.adjust(mini_comp_stats$hr_pval, method = "BY", n = 10)
#mini_comp_stats$hr_q_BH <- p.adjust(mini_comp_stats$hr_pval, method = "BH", n = 10)
mini_comp_stats$hr_q_Bonferroni <- p.adjust(mini_comp_stats$hr_pval, method = "bonferroni", n = 10)
mini_comp_stats$hr_q_FDR <- p.adjust(mini_comp_stats$hr_pval, method = "fdr", n = 10)
mini_comp_stats$hr_q_holm <- p.adjust(mini_comp_stats$hr_pval, method = "holm", n = 10)

#Prepare the data for plotting by flipping wide -> long, creating one row per variable per q.
#This also creates a "color_guide" variable in the form of significance

# extraversion, open_mindedness, term_GPA, UM_credits_at_study, semester_study,
# Goal_grade_sd, grade_100_mean, grade_minus_goal_mean, NA_dense_mean, pred_100_mean 

plot_data_pq <- mini_comp_stats |>
  filter(variable %in% MVF) |>
  select(
    variable, hazard_ratio,
    hr_pval,
    hr_q_BY, hr_q_Bonferroni, hr_q_FDR, hr_q_holm
  ) |>
  pivot_longer(
    cols = c(hr_pval, hr_q_BY, hr_q_Bonferroni, hr_q_FDR, hr_q_holm),
    names_to = "stat_type",
    values_to = "pval"
  ) |>
  mutate(
    method = case_when(
      stat_type == "hr_pval" ~ "Raw P",
      stat_type == "hr_q_BY" ~ "BY",
      stat_type == "hr_q_Bonferroni" ~ "Bonferroni",
      stat_type == "hr_q_FDR" ~ "FDR",
      stat_type == "hr_q_holm" ~ "Holm"
      ), 
    variable = case_when(
      variable == "extraversion" ~ "Extraversion",
      variable == "open_mindedness" ~ "Open Mindedness",
      variable == "term_GPA" ~ "GPA Achieved during Semester of Study Participation",
      variable == "UM_credits_at_study" ~ "Number of Credits Earned in UM at time of Study Participation",
      variable == "semester_study" ~ "Number of Semesters in UM at time of Study",
      variable == "Goal_grade_sd" ~ "Standard Deviation of Goal Grade",
      variable == "grade_100_mean" ~ "Mean Grade",
      variable == "grade_minus_goal_mean" ~ "Mean Grade minus Goal",
      variable == "NA_dense_mean" ~ "Mean Dense Period Negative Affect",
      variable == "pred_100_mean" ~ "Mean Predicted Grade",
    ),
    significance = if_else(pval < 0.05, if_else(pval < 0.01, if_else(pval < 0.001, "SIG001", "SIG01"), "SIG05"), "NSIG"),
    variable = fct_reorder(variable, pval, .fun = min)
  )

hr_labels <- plot_data_pq |>
  filter(method == "Raw P") |>
  distinct(variable, hazard_ratio)


#####Multiple Comparisons Corrected P-values Plot#####
ggplot(plot_data_pq, aes(x = pval, y = variable)) +
  geom_point(
    aes(shape = method, color = significance),
    size = 4#,
    #position = position_dodge(width = 0.7)
  ) + #Label HRs
  geom_text(
    data = hr_labels,
    aes(x = 0.0005, label = paste0("HR = ", round(hazard_ratio, 2))),
    y = hr_labels$variable,
    vjust = 2,
    hjust = 8.7,
    size = 4.8,
    inherit.aes = FALSE
  ) + #Create Reference Lines and Label Them
  geom_vline(xintercept = .05, linetype = "dashed", colour = "black",linewidth = 1) +
  geom_vline(xintercept = .01, linetype = "dashed", colour = "blue",linewidth = 1) +
  geom_vline(xintercept = .001, linetype = "dashed", colour = "red",linewidth = 1) +
  annotate("text", x = 0.05, y = Inf, label = "p = .05", vjust = 1, hjust = 1, size = 5, colour = "black") +
  annotate("text", x = 0.01, y = Inf, label = "p = .01", vjust = 1, hjust = 1, size = 5, color = "blue") +
  annotate("text", x = 0.001, y = Inf, label = "p = .001", vjust = 1, hjust = 1, size = 5, color = "red") +
  #Log Scale The X to make it easier to read
  scale_x_log10() + #Color the dots to make it even easier to read
  scale_color_manual(
    values = c("SIG001" = "red", "SIG01" = "blue", "SIG05" = "black", "NSIG" = "gray60"),
    guide = "none"
  ) +
  scale_shape_manual(
    values = c("Raw P" = 19, "BY" = 17, "Bonferroni" = 15, "FDR" = 18, "Holm" = 3)
  ) +
  labs(
    x = "p values and adjusted p-values",
    y = NULL,
    shape = "Adjustment Method",
    title = "P-values and Adjusted P-values for STEM Dropout Predictors",
    #subtitle = "Hazard Ratio shown below each point; Significance based on p < .05"
  ) +
  scale_y_discrete(limits = rev) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 14),
    axis.title.x = element_text(size = 17)
  )

#####Survival Surfaces#####
#This code will create the object and then plot it but calling the object. All of the graphs in the paper can be found below.

##Interactive 3D surface for mean_GMG

#Load the relevant statistics
cox_model <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ grade_minus_goal_mean, data = sobreviver_comp,  x = TRUE)
summary_obj <- summary(cox_model)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]
bon <- p.adjust(p_val, method = "bonferroni", n = 10)


#Set up the plot call
plot_GMG <-  plot_surv_3Dsurface(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "grade_minus_goal_mean",
                                 data = sobreviver_comp,
                                 model = cox_model,
                                 interactive=T,
                                 max_t = 10,
                                 zlab = "Mean Grade Minus Goal",
                                 xlab = "Time (Semesters)",
                                 ylab = 'Probability of STEM Persistence')

#add a title
plot_GMG <- plot_GMG |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of the Mean Grade minus Goal",
      #"<br><sub>",
      #"Cox Model q = ", formatC(bon, format = "f", digits = 10),
      #" | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    y= 0.9,
    xanchor = "center"
  )
)

#Plot in Rstudio viewer. Now, you're going to have to manually move the view to find the best view, and then save as png, unfortunately, saving as svg
#Is not exactly available
plot_GMG

##Interactive 3D surface for NA_dense_mean
cox_model <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ NA_dense_mean, data = sobreviver_comp,  x = TRUE)
summary_obj <- summary(cox_model)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]
bon <- p.adjust(p_val, method = "bonferroni", n = 10)

plot_dNA <-  plot_surv_3Dsurface(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "NA_dense_mean",
                                 data = sobreviver_comp,
                                 model = cox_model,
                                 interactive=T,
                                 max_t = 10,
                                 zlab = "Mean Dense Period Negative Affect",
                                 xlab = "Time (Semesters)",
                                 ylab = 'Probability of STEM Persistence')

plot_dNA <- plot_dNA |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of the Mean Dense Period Negative Affect", 
      # "<br><sub>",
      # "Cox Model q = ", formatC(bon, format = "f", digits = 10),
      # " | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    y= 0.9,
    xanchor = "center"
  )
)

plot_dNA

##Interactive 3D surface for mean PE
cox_model <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ pred_100_mean, data = sobreviver_comp,  x = TRUE)
summary_obj <- summary(cox_model)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]
bon <- p.adjust(p_val, method = "bonferroni", n = 10)

plot_PE <-  plot_surv_3Dsurface(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "pred_100_mean",
                                 data = sobreviver_comp,
                                 model = cox_model,
                                 interactive=T,
                                 max_t = 10,
                                 zlab = "Mean Predicted Grade",
                                 xlab = "Time (Semesters)",
                                 ylab = 'Probability of STEM Persistence')

plot_PE <- plot_PE |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of the Mean Predicted Grade", 
      # "<br><sub>",
      # "Cox Model q = ", formatC(bon, format = "f", digits = 10),
      # " | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    y= 0.9,
    xanchor = "center"
  )
)

plot_PE

##Interactive 3D surface for mean Grade
cox_model <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ grade_100_mean, data = sobreviver_comp,  x = TRUE)
summary_obj <- summary(cox_model)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]
bon <- p.adjust(p_val, method = "bonferroni", n = 10)

plot_grd <-  plot_surv_3Dsurface(time = "semester_drop",
                                status = "tier1_STEM_drop_tozero",
                                variable = "grade_100_mean",
                                data = sobreviver_comp,
                                model = cox_model,
                                interactive=T,
                                max_t = 10,
                                zlab = "Mean Grade",
                                xlab = "Time (Semesters)",
                                ylab = 'Probability of STEM Persistence')

plot_grd <- plot_grd |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of the Mean Grade", 
      # "<br><sub>",
      # "Cox Model q = ", formatC(bon, format = "f", digits = 10),
      # " | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    y= 0.9,
    xanchor = "center"
  )
)

plot_grd

##Interactive 3D surface for term_GPA
cox_model <- coxph(Surv(time = semester_drop, event = tier1_STEM_drop_tozero) ~ term_GPA, data = sobreviver_comp,  x = TRUE)
summary_obj <- summary(cox_model)
p_val <- summary_obj$coefficients[, "Pr(>|z|)"]
HR <- summary_obj$coefficients[, "exp(coef)"]
bon <- p.adjust(p_val, method = "bonferroni", n = 10)

plot_tGPA <-  plot_surv_3Dsurface(time = "semester_drop",
                                 status = "tier1_STEM_drop_tozero",
                                 variable = "term_GPA",
                                 data = sobreviver_comp,
                                 model = cox_model,
                                 interactive=T,
                                 max_t = 10,
                                 zlab = "Mean GPA Achieved during Semester of Study Participation",
                                 xlab = "Time (Semesters)",
                                 ylab = 'Probability of STEM Persistence')

plot_tGPA <- plot_tGPA |> layout(
  title = list(
    text = paste0(
      "STEM Persistence Probability over Time as a Function of the Mean GPA Achieved during Semester of Study Participation", 
      # "<br><sub>",
      # "Cox Model q = ", formatC(bon, format = "f", digits = 10),
      # " | Cox HR = ", sprintf("%.5f", HR),
      "</sub>"
    ),
    x = 0.5,
    y= 0.9,
    xanchor = "center"
  )
)

plot_tGPA

