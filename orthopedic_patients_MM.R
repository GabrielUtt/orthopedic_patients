# -------------------- PURPOSE --------------------
# This is the R-code for a data-science capstone project, applying
# various (machine learning) algorithms to a set of data about orthopedic
# patients. The results are presented in a separate .Rmd document
# The code in this script is organized in sections that do not
# in general reflect the thought process I went through when doing this project.
# For example, all data are prepared in one section of the R-code, while
# when working on the project, I first worked with models on the original data,
# and only later tried to normalize the variables.
# The though process of how this project was approached is reflected in the
# accompanying .Rmd document

# -------------------- PREREQUISITS/INPUTS --------------------

# None: The necessary input data can be provided as .csv files in a subfolder
# called /data, but if not, the data are provided via a separate R-script,
# called "help_functions_and_data.R".

## ---- load_libraries ----
# functionality
if (!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)
if (!require(caret)) install.packages("caret")
library(caret) # for process of using predictive models
if (!require(rpart)) install.packages("rpart")
library(rpart) # to use regression trees
if (!require(cvms)) install.packages("cvms")
library(cvms) # cross validation for model selection
# depiction
if (!require(gridExtra)) install.packages("gridExtra")
library(gridExtra) # to arrange multiple plots
if (!require(ggcorrplot)) install.packages("ggcorrplot")
library(ggcorrplot) # to depict correlation plot
if (!require(rpart.plot)) install.packages("rpart.plot")
library(rpart.plot) # to depict regression tree selection criteria
if (!require(kableExtra)) install.packages("kableExtra")
library(kableExtra) # to add functionality to knitr::kable() tables

Sys.setenv(LANG = "en") # set system language to English

## ---- get_data_and_help_functions ----
# providing two data sets with identical observations and differing definition
# of outcome (two_cat vs three_cat) for orthopedic patients

source("help_functions_and_data.R")

## __normalize_data ----
# normalized data sets (method 1) - all explanatory normalized, appendix _norm1
two_cat_pat_norm1 <- two_cat_pat %>%
  mutate(across(.cols = -c(class, pat_Id), .fns = ~ c(scale(.))))

three_cat_pat_norm1 <- three_cat_pat %>%
  mutate(across(.cols = -c(class, pat_Id), .fns = ~ c(scale(.))))

# normalized data sets (method 2) - files with appendix '_norm2'
# with method 1, linear relationship PI = PT + SS does not hold anymore
# with method 2, linear relationship PI = PT + SS is maintained,
# with PI being normalized
two_cat_pat_norm2 <- two_cat_pat %>%
  mutate(
    across(.cols = -c(class, pat_Id, pelvic_tilt, sacral_slope),
           .fns = ~ c(scale(.)))) %>%
  mutate(
    PT_norm = pelvic_tilt / (pelvic_tilt + sacral_slope),
    SS_norm = sacral_slope / (pelvic_tilt + sacral_slope)
  ) %>%
  select(-c(pelvic_tilt, sacral_slope)) %>%
  rename(
    pelvic_tilt = PT_norm,
    sacral_slope = SS_norm
  )

three_cat_pat_norm2 <- three_cat_pat %>%
  mutate(across(
    .cols = -c(class, pat_Id, pelvic_tilt, sacral_slope),
    .fns = ~ c(scale(.)))) %>%
  mutate(
    PT_norm = pelvic_tilt / (pelvic_tilt + sacral_slope),
    SS_norm = sacral_slope / (pelvic_tilt + sacral_slope)
  ) %>%
  select(-c(pelvic_tilt, sacral_slope)) %>%
  rename(
    pelvic_tilt = PT_norm,
    sacral_slope = SS_norm
  )

## __create_train_and_test_sets ----
set.seed(2022)
test_ind <- createDataPartition(two_cat_pat$class, p = 0.3, list = FALSE)

test2_set <- two_cat_pat[test_ind, ]
test2_set_norm <- two_cat_pat_norm1[test_ind, ]
test2_set_norm2 <- two_cat_pat_norm2[test_ind, ]
train2_set <- two_cat_pat[-test_ind, ]
train2_set_norm <- two_cat_pat_norm1[-test_ind, ]
train2_set_norm2 <- two_cat_pat_norm2[-test_ind, ]

test3_set <- three_cat_pat[test_ind, ]
test3_set_norm <- three_cat_pat_norm1[test_ind, ]
test3_set_norm2 <- three_cat_pat_norm2[test_ind, ]
train3_set <- three_cat_pat[-test_ind, ]
train3_set_norm <- three_cat_pat_norm1[-test_ind, ]
train3_set_norm2 <- three_cat_pat_norm2[-test_ind, ]

## __data_sets_for_exploratory_analysis ----

# create longer data sets for exploratory analysis (plots etc.)
# Plots will be directly created in .Rmd document
train2_set_long <- train2_set %>% pivot_longer(
  cols = all_of(setdiff(names(train2_set), c("pat_Id", "class")))
)

train2_set_norm_long <- train2_set_norm %>% pivot_longer(
  cols = all_of(setdiff(names(train2_set), c("pat_Id", "class")))
)

test2_set_long <- test2_set %>% pivot_longer(
  cols = all_of(setdiff(names(train2_set), c("pat_Id", "class")))
)

test2_set_norm_long <- test2_set_norm %>% pivot_longer(
  cols = all_of(setdiff(names(train2_set), c("pat_Id", "class")))
)

train3_set_long <- train3_set %>% pivot_longer(
  cols = all_of(setdiff(names(train3_set), c("pat_Id", "class")))
)

train3_set_norm_long <- train3_set_norm %>% pivot_longer(
  cols = all_of(setdiff(names(train3_set), c("pat_Id", "class")))
)

test3_set_long <- test3_set %>% pivot_longer(
  cols = all_of(setdiff(names(train3_set), c("pat_Id", "class")))
)

test3_set_norm_long <- test3_set_norm %>% pivot_longer(
  cols = all_of(setdiff(names(train3_set), c("pat_Id", "class")))
)

## ---- Models_binary ----

## __linear_regression ----

# Linear Regression
fit_lm <- train2_set %>%
  mutate(y = as.numeric(class == "Abnormal")) %>%
  select(-c("class", "pelvic_inc", "pat_Id")) %>% # pelvic_inc a linear comb
  glm(y ~ ., data = .)

p_hat_lm <- predict(fit_lm, newdata = test2_set) # ~ probability prediction
# category prediction
y_hat_lm <- factor( 
  ifelse(p_hat_lm > 0.5, "Abnormal", "Normal")
  )

model_results <- 
  f_save_model_results("linear regression", test2_set, y_hat_lm, beta_F = 3)

# False Positives with linear regression
FP_lm <- test2_set$pat_Id[y_hat_lm == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with linear regression
FN_lm <- test2_set$pat_Id[y_hat_lm == "Normal" & test2_set$class == "Abnormal"]

## __logistic_regression ----

# Logistic Regression
fit_glm <- train2_set %>%
  mutate(y = as.numeric(class == "Abnormal")) %>%
  select(-c("class", "pelvic_inc", "pat_Id")) %>% # pelvic_inc a linear comb
  glm(y ~ ., data = ., family = "binomial")

# probability prediction
p_hat_glm <- predict(fit_glm, newdata = test2_set, type = "response")
# category prediction
y_hat_glm <- factor(ifelse(p_hat_glm > 0.5, "Abnormal", "Normal")) 

model_results <- 
  f_save_model_results("logistic regression", test2_set, y_hat_glm, beta_F = 3)

# False Positives with logistic regression
FP_glm <- 
  test2_set$pat_Id[y_hat_glm == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with logistic regression
FN_glm <- 
  test2_set$pat_Id[y_hat_glm == "Normal" & test2_set$class == "Abnormal"]

## __cross_validation_parameters ----

# using the same cross validation parameters for all machine-learning approaches
k_cv <- 20 # k-fold cross validation
p_cv <- 0.8 # size of train set in cross validation

control <- trainControl(method = "cv", number = k_cv, p = p_cv)

## __regression_tree ----

set.seed(1)
# train regression tree
train_rpart <- train(class ~ .,
  method = "rpart",
  tuneGrid = data.frame(cp = seq(0.001, 0.08, 0.001)),
  data = train2_set %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train_rpart)

# use full train set to fit "finalModel" from cross validation
fit_rpart <- rpart(class ~ .,
  data = train2_set %>%
    select(-c("pelvic_inc", "pat_Id")),
  control = rpart.control(cp = train_rpart$bestTune)
)

# probability prediction
p_hat_rpart <- predict(fit_rpart, newdata = test2_set)[, "Abnormal"]
# category prediction
y_hat_rpart <- predict(fit_rpart, newdata = test2_set, type = "class")

model_results <- 
  f_save_model_results("regression tree", test2_set, y_hat_rpart, beta_F = 3)

# False Positives with regression tree
FP_rpart <- 
  test2_set$pat_Id[y_hat_rpart == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with regression tree
FN_rpart <- 
  test2_set$pat_Id[y_hat_rpart == "Normal" & test2_set$class == "Abnormal"]

## __loess ----

grid <- expand.grid(span = seq(0.05, 0.95, 0.05), degree = 1)
set.seed(1)
train_loess <- train(class ~ .,
  method = "gamLoess",
  tuneGrid = grid,
  data = train2_set %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train_loess)

## loess() function does not work with more than 4 explanatory variables, 
## unclear why train with "gamLoess" works
## therefore, finalModel from loess cross validation cannot be fitted with 
## full train set
# fit_loess <- loess(
#   class ~ ., 
#   degree = 1, 
#   span = train_loess$finalModel$tuneValue,
#   data = train2_set %>% select(-c("pelvic_inc", "pat_Id")))

# probability prediction
p_hat_loess <- predict(train_loess, newdata = test2_set, type = "prob")
# category prediction
y_hat_loess <- predict(train_loess, newdata = test2_set) 

model_results <- 
  f_save_model_results("loess", test2_set, y_hat_loess, beta_F = 3)

# False Positives with loess
FP_loess <- 
  test2_set$pat_Id[y_hat_loess == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with loess
FN_loess <- 
  test2_set$pat_Id[y_hat_loess == "Normal" & test2_set$class == "Abnormal"]

## __knn ----

set.seed(1)
train_knn <- train(class ~ .,
  method = "knn",
  tuneGrid = data.frame(k = seq(3, 45, 2)),
  data = train2_set %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train_knn)

# use full train set to fit "finalModel" from cross validation
fit_knn <- knn3(class ~ ., train2_set %>%
  select(-c("pelvic_inc", "pat_Id")), k = train_knn$finalModel[["k"]])

# probability prediction
p_hat_knn <- predict(fit_knn, newdata = test2_set, type = "prob")[, "Abnormal"]
# category prediction
y_hat_knn <- predict(fit_knn, newdata = test2_set, type = "class") 

model_results <- 
  f_save_model_results("knn", test2_set, y_hat_knn, beta_F = 3)

# False Positives with knn
FP_knn <- 
  test2_set$pat_Id[y_hat_knn == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with knn
FN_knn <- 
  test2_set$pat_Id[y_hat_knn == "Normal" & test2_set$class == "Abnormal"]

## __knn_normalized ----

set.seed(1)
train_knn_norm <- train(class ~ .,
  method = "knn",
  tuneGrid = data.frame(k = seq(3, 45, 2)),
  data = train2_set_norm %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train_knn_norm)

# use full train set to fit "finalModel" from cross validation
fit_knn_norm <- knn3(class ~ ., train2_set_norm %>%
  select(-c("pelvic_inc", "pat_Id")), k = train_knn_norm$finalModel[["k"]])

# probability prediction
p_hat_knn_norm <- 
  predict(fit_knn_norm, newdata = test2_set_norm, type = "prob")[, "Abnormal"]
# category prediction
y_hat_knn_norm <- 
  predict(fit_knn_norm, newdata = test2_set_norm, type = "class") 

model_results <- 
  f_save_model_results("knn_norm", test2_set_norm, y_hat_knn_norm, beta_F = 3)

# when normalizing, linear relationship PI = PT + SS does not hold anymore!
train2_set_norm %>%
  mutate(add = pelvic_tilt + sacral_slope) %>%
  ggplot() +
  geom_line(aes(x = pat_Id, y = pelvic_inc, color = "blue")) +
  geom_line(aes(x = pat_Id, y = add, color = "red"))

# False Positives with knn_norm
FP_knn_norm <- 
  test2_set$pat_Id[y_hat_knn_norm == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with knn_norm
FN_knn_norm <- 
  test2_set$pat_Id[y_hat_knn_norm == "Normal" & test2_set$class == "Abnormal"]


## __knn_normalized_ver2 ----


# using data set where PI is normalized and linear relation PI = PT + SS holds
set.seed(1)
train_knn_norm2 <- train(class ~ .,
  method = "knn",
  tuneGrid = data.frame(k = seq(3, 45, 2)),
  data = train2_set_norm2 %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train_knn_norm2)

# use full train set to fit "finalModel" from cross validation
fit_knn_norm2 <- knn3(class ~ ., train2_set_norm2 %>%
  select(-c("pelvic_inc", "pat_Id")), k = train_knn_norm2$finalModel[["k"]])

# probability prediction
p_hat_knn_norm2 <- 
  predict(fit_knn_norm2, newdata = test2_set_norm2, type = "prob")[, "Abnormal"]
# category prediction
y_hat_knn_norm2 <- 
  predict(fit_knn_norm2, newdata = test2_set_norm2, type = "class") 

model_results <- 
  f_save_model_results("knn_norm2", test2_set_norm2, y_hat_knn_norm2, beta_F = 3)

# False Positives with knn_norm2
FP_knn_norm2 <- 
  test2_set$pat_Id[y_hat_knn_norm2 == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with knn_norm2
FN_knn_norm2 <- 
  test2_set$pat_Id[y_hat_knn_norm2 == "Normal" & test2_set$class == "Abnormal"]

## __combination_models ----

# # combination of knn_norm2 and linear regression - not used in final paper
# p_hat_comb1 = rowMeans(cbind(p_hat_lm, p_hat_knn_norm2))
# y_hat_comb1 <- factor(ifelse(p_hat_comb1 > 0.5, "Abnormal", "Normal"))
#
# confusionMatrix(y_hat_comb1, test2_set$class)$table
# confusionMatrix(y_hat_comb1, test2_set$class)$byClass
#
# model_results <-  
#   f_save_model_results(
# "p_comb(lm, knn_norm2)", test2_set, y_hat_comb1, beta_F = 3
# )

# combination of all approaches (except inferior knn models) / wghtd prob
# average probability prediction
p_hat_comb_full <- 
  rowMeans(cbind(
    p_hat_lm, 
    p_hat_glm, 
    p_hat_knn_norm2, 
    p_hat_loess, 
    p_hat_rpart)
    ) 

# outcome prediction
y_hat_comb_full <- 
  factor(ifelse(p_hat_comb_full > 0.50, "Abnormal", "Normal")) 

model_results <- 
  f_save_model_results("p_comb_bin", test2_set, y_hat_comb_full, beta_F = 3)

# False Positives with p_comb_bin
FP_p_comb_bin <- 
  test2_set$pat_Id[y_hat_comb_full == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with p_comb_bin
FN_p_comb_bin <- 
  test2_set$pat_Id[y_hat_comb_full == "Normal" & test2_set$class == "Abnormal"]

# comb of all approaches (except inferior knn models) / majority prediction
y_hat_comb_maj_full <- 
  factor(
    if_else(
      (y_hat_lm == "Abnormal") + (y_hat_glm == "Abnormal") + 
        (y_hat_rpart == "Abnormal") + (y_hat_knn_norm2 == "Abnormal") + 
        (y_hat_loess == "Abnormal") > 2.5,
      "Abnormal", 
      "Normal"
    )
  )

model_results <- 
  f_save_model_results("maj_comb_bin", test2_set, y_hat_comb_maj_full, beta_F = 3)

# False Positives with maj_comb_bin
FP_maj_comb_bin <- 
  test2_set$pat_Id[y_hat_comb_maj_full == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with maj_comb_bin
FN_maj_comb_bin <- 
  test2_set$pat_Id[y_hat_comb_maj_full == "Normal" & test2_set$class == "Abnormal"]

## ---- Models_categorical ----

## __regression_tree (3-cat) ----

set.seed(1)
train3_rpart <- train(class ~ .,
  method = "rpart",
  tuneGrid = data.frame(cp = seq(0.001, 0.08, 0.001)),
  data = train3_set %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train3_rpart)

# use full train set to fit "finalModel" from cross validation
fit3_rpart <- rpart(class ~ .,
  data = train3_set %>%
    select(-c("pelvic_inc", "pat_Id")),
  control = rpart.control(cp = train3_rpart$bestTune)
)

# probability prediction
p_hat3_rpart <- predict(fit3_rpart, newdata = test3_set)
# outcome prediction
y_hat3_rpart <- predict(fit3_rpart, newdata = test3_set, type = "class") 
y_hat3_rpart_bin <- 
  factor(
    str_replace(y_hat3_rpart, "Hernia|Spondylolisthesis", "Abnormal"),
    levels = c("Abnormal", "Normal")
  ) # translate back to binary outcome (for comparison purposes)

model_results <- f_save_model_results(
  "rpart_cat",
  test3_set %>% mutate(
    class = factor(str_replace(class, "Hernia|Spondylolisthesis", "Abnormal"),
    levels = c("Abnormal", "Normal")
    )),
  y_hat3_rpart_bin,
  beta_F = 3
)

cm_3rpart <- confusionMatrix(y_hat3_rpart, test3_set$class)

# False Positives with regression tree (3-cat)
FP_3rpart <- 
  test2_set$pat_Id[y_hat3_rpart_bin == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with regression tree (3-cat)
FN_3rpart <- 
  test2_set$pat_Id[y_hat3_rpart_bin == "Normal" & test2_set$class == "Abnormal"]

## __knn_cat ----

set.seed(1)
train3_knn <- train(class ~ .,
  method = "knn",
  tuneGrid = data.frame(k = seq(3, 45, 2)),
  data = train3_set %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train3_knn)

# use full train set to fit "finalModel" from cross validation
fit3_knn <- knn3(class ~ ., train3_set %>%
  select(-c("pelvic_inc", "pat_Id")), k = train3_knn$finalModel[["k"]])

# probability prediction
p_hat3_knn <- predict(fit3_knn, newdata = test3_set)
# outcome prediction
y_hat3_knn <- predict(fit3_knn, newdata = test3_set, type = "class") 
y_hat3_knn_bin <- factor(
  str_replace(y_hat3_knn, "Hernia|Spondylolisthesis", "Abnormal"),
  levels = c("Abnormal", "Normal")
) # translate back to binary outcome (for comparison purposes)

model_results <- f_save_model_results(
  "knn_cat",
  test3_set %>% 
    mutate(class = factor(
      str_replace(class, "Hernia|Spondylolisthesis", "Abnormal"),
      levels = c("Abnormal", "Normal")
  )),
  y_hat3_knn_bin,
  beta_F = 3
)

cm_3knn <- confusionMatrix(y_hat3_knn, test3_set$class)
cm_3knn$table

# False Positives with knn (3-cat)
FP_3knn <- 
  test2_set$pat_Id[y_hat3_knn_bin == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with knn (3-cat)
FN_3knn <- 
  test2_set$pat_Id[y_hat3_knn_bin == "Normal" & test2_set$class == "Abnormal"]

## __knn_normalized (3cat) ----

set.seed(1)
train3_knn_norm <- train(class ~ .,
  method = "knn",
  tuneGrid = data.frame(k = seq(3, 45, 2)),
  data = train3_set_norm %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train3_knn_norm)

# use full train set to fit "finalModel" from cross validation
fit3_knn_norm <- knn3(class ~ ., train3_set_norm %>%
  select(-c("pelvic_inc", "pat_Id")), k = train3_knn_norm$finalModel[["k"]])

# probability prediction
p_hat3_knn_norm <- predict(fit3_knn_norm, newdata = test3_set_norm)
# outcome prediction
y_hat3_knn_norm <- 
  predict(fit3_knn_norm, newdata = test3_set_norm, type = "class") 
y_hat3_knn_norm_bin <- factor(
  str_replace(y_hat3_knn_norm, "Hernia|Spondylolisthesis", "Abnormal"),
  levels = c("Abnormal", "Normal")
) # translate back to binary outcome (for comparison purposes)

model_results <- f_save_model_results(
  "knn_norm_cat",
  test3_set_norm %>% 
    mutate(class = factor(
      str_replace(class, "Hernia|Spondylolisthesis", "Abnormal"),
      levels = c("Abnormal", "Normal")
  )),
  y_hat3_knn_norm_bin,
  beta_F = 3
)

cm_3knn_norm2 <- confusionMatrix(y_hat3_knn_norm, test3_set_norm$class)

# False Positives with knn (3-cat)
FP_3knn_norm <- 
  test2_set$pat_Id[y_hat3_knn_norm_bin == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with knn (3-cat)
FN_3knn_norm <- 
  test2_set$pat_Id[y_hat3_knn_norm_bin == "Normal" & test2_set$class == "Abnormal"]

## __knn_normalized_ver2 (3cat) ----

set.seed(1)
train3_knn_norm2 <- train(class ~ .,
  method = "knn",
  tuneGrid = data.frame(k = seq(3, 45, 2)),
  data = train3_set_norm2 %>%
    select(-c("pelvic_inc", "pat_Id")),
  trControl = control
)

plot(train3_knn_norm2)

# use full train set to fit "finalModel" from cross validation
fit3_knn_norm2 <- knn3(class ~ ., train3_set_norm2 %>%
  select(-c("pelvic_inc", "pat_Id")), k = train3_knn_norm2$finalModel[["k"]])

p_hat3_knn_norm2 <- predict(fit3_knn_norm2, newdata = test3_set_norm2)
y_hat3_knn_norm2 <- 
  predict(fit3_knn_norm2, newdata = test3_set_norm2, type = "class")
y_hat3_knn_norm2_bin <- 
  factor(
    str_replace(y_hat3_knn_norm2, "Hernia|Spondylolisthesis", "Abnormal"),
    levels = c("Abnormal", "Normal")
)

model_results <- f_save_model_results(
  "knn_norm2_cat",
  test3_set_norm2 %>% 
    mutate(class = factor(
      str_replace(class, "Hernia|Spondylolisthesis", "Abnormal"),
      levels = c("Abnormal", "Normal")
  )),
  y_hat3_knn_norm2_bin,
  beta_F = 3
)

cm_3knn_norm2 <- confusionMatrix(y_hat3_knn_norm2, test3_set_norm2$class)

# False Positives with knn (3-cat)
FP_3knn_norm2 <- 
  test2_set$pat_Id[y_hat3_knn_norm2_bin == "Abnormal" & test2_set$class == "Normal"]
# False Negatives with knn (3-cat)
FN_3knn_norm2 <- 
  test2_set$pat_Id[y_hat3_knn_norm2_bin == "Normal" & test2_set$class == "Abnormal"]

## __combination rpart + knn_cat ----

p_hat3_comb1 <- (p_hat3_knn + p_hat3_rpart) / 2 # weighted probability
y_hat3_comb1 <- factor(colnames(p_hat3_comb1)[apply(p_hat3_comb1, 1, which.max)],
  levels = c("Hernia", "Spondylolisthesis", "Normal")
) # category prediction, column with maximum prob
y_hat3_comb1_bin <- factor(
  str_replace(y_hat3_comb1, "Hernia|Spondylolisthesis", "Abnormal"),
  levels = c("Abnormal", "Normal")
)

model_results <- f_save_model_results(
  "p_comb1_cat",
  test3_set_norm2 %>% 
    mutate(class = factor(
      str_replace(class, "Hernia|Spondylolisthesis", "Abnormal"),
      levels = c("Abnormal", "Normal")
  )),
  y_hat3_comb1_bin,
  beta_F = 3
)

cm_3comb1 <- confusionMatrix(y_hat3_comb1, test3_set$class)

# False Positives with comb rpart, knn_cat
FP_3comb1 <-
  test2_set$pat_Id[y_hat3_comb1_bin == "Abnormal" & test2_set$class == "Normal"]
# False Positives with comb rpart, knn_cat
FN_3comb1 <-
  test2_set$pat_Id[y_hat3_comb1_bin == "Normal" & test2_set$class == "Abnormal"]

## __combination rpart + knn_norm2 (3-cat) ----

# weighted probability
p_hat3_comb2 <- (as.matrix(p_hat3_knn_norm2) + as.matrix(p_hat3_rpart)) / 2 
y_hat3_comb2 <- factor(
  colnames(p_hat3_comb2)[apply(p_hat3_comb2, 1, which.max)],
  levels = c("Hernia", "Spondylolisthesis", "Normal")
) # category prediction, column with maximum prob
y_hat3_comb2_bin <- factor(
  str_replace(y_hat3_comb2, "Hernia|Spondylolisthesis", "Abnormal"),
  levels = c("Abnormal", "Normal")
)

model_results <- f_save_model_results(
  "p_comb2_cat",
  test3_set_norm2 %>% 
    mutate(class = factor(
      str_replace(class, "Hernia|Spondylolisthesis", "Abnormal"),
      levels = c("Abnormal", "Normal")
  )),
  y_hat3_comb2_bin,
  beta_F = 3
)

cm_3comb2 <- confusionMatrix(y_hat3_comb2, test3_set$class)

# False Positives with comb rpart, knn_cat
FP_3comb2 <- 
  test2_set$pat_Id[y_hat3_comb2_bin == "Abnormal" & test2_set$class == "Normal"]
# False Positives with comb rpart, knn_cat
FN_3comb2 <- 
  test2_set$pat_Id[y_hat3_comb2_bin == "Normal" & test2_set$class == "Abnormal"]

## ---- __missed_by_every_algorithm ----

FP_all_models <- Reduce(intersect, list(
  FP_lm, FP_glm, FP_rpart, FP_loess, FP_knn, FP_knn_norm, FP_knn_norm2, 
  FP_p_comb_bin, FP_maj_comb_bin, FP_3rpart, FP_3knn, FP_3knn_norm, 
  FP_3knn_norm2, FP_3comb1, FP_3comb2
))

FN_all_models <- Reduce(intersect, list(
  FN_lm, FN_glm, FN_rpart, FN_loess, FN_knn, FN_knn_norm, FN_knn_norm2, 
  FN_p_comb_bin, FN_maj_comb_bin, FN_3rpart, FN_3knn, FN_3knn_norm, 
  FN_3knn_norm2, FN_3comb1, FN_3comb2
))
