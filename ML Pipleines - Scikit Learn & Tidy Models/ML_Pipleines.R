
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------- Building models with Tidy Model workflows -------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library(tidymodels)
library(tidyverse)
library(workflows)
library(tune)
library(nnet)
library(workflowsets)

#-----------------------------------------------------------------------------------
# Load the data
#-----------------------------------------------------------------------------------

df <- read.csv('hcvdata.csv')


#-----------------------------------------------------------------------------------
# Exploratory data analysis
#-----------------------------------------------------------------------------------

summary(df)


#-----------------------------------------------------------------------------------
# Splitting the data
#-----------------------------------------------------------------------------------

hcv_split <- initial_split(df, prop = 3/4)

# extract training and testing sets
hcv_train <- training(hcv_split)
hcv_test <- testing(hcv_split)



numeric_features <- df %>% 
  select_if(is.numeric)
names(numeric_features)


categorical_features <- df %>% 
  select_if(is.character) %>% 
  select(Sex)

names(categorical_features)

#-----------------------------------------------------------------------------------
# define the recipe
#-----------------------------------------------------------------------------------
hcv_recipe <- 
  # which consists of the formula (outcome ~ predictors)
  recipe(Category ~ Age + Sex + ALB + ALP + ALT +AST +BIL+CHE+CHOL+CREA+GGT+PROT, 
         data = df)
  # and some pre-processing steps
  step_normalize(all_numeric()) %>%
  step_knnimpute(all_predictors())
  
#-----------------------------------------------------------------------------------
# define model specification
#-----------------------------------------------------------------------------------

glm_spec <-
  multinom_reg(mode = "classification") %>%
  set_engine("nnet")


rpart_spec <-
  decision_tree(mode = "classification") %>%
  set_engine("rpart")

  
#-----------------------------------------------------------------------------------
# define workflow
#-----------------------------------------------------------------------------------
hcv_wf <- workflow() %>%
  add_recipe(hcv_recipe) %>%
  add_model(glm_spec)


#-----------------------------------------------------------------------------------
# Model metrics
#-----------------------------------------------------------------------------------

final_res <- last_fit(hcv_wf, split = hcv_split)

collect_metrics(final_res)


#-----------------------------------------------------------------------------------
# Generate confusion matrix
#-----------------------------------------------------------------------------------

collect_predictions(final_res) %>% 
  conf_mat(Category, .pred_class)


  




