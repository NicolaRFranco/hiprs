#####################################################################################
###############         Glinternet (Lim and Hastie, 2015)             ###############
###############               benchmark experiment                    ###############
#####################################################################################


#### REQUIRED PACKAGES 
library(devtools)
#devtools::install_github("karlkumbier/iRF2.0")
library(stringr)
library(dplyr)
library(readr)
library(caret) 
#install.packages("dummies")
library(dummies)
#install.packages("glinternet")
library(glinternet)
library(pROC)
#install.packages("RJSONIO")
library(RJSONIO)
library(yardstick)


# NOTE ON SIMULATED DATA FOR BENCHMARK EXPERIMENTS:
# Simulation data are generated through our custom python function 
# (cf. Behravan Benchmark Experiment notebook) and stored in a defined folder.
# Set working directory to this folder so that this script can pick them and 
# run the experiment on the same data.

setwd("C:/Users/Michela/LIPS 2.0")

# Set some useful variables before starting the loop experiment
n_simulations <- 30 - 1

RESULTS <- data.frame()
position = 1

# Manually set this range of experiments
Number_of_Interactions = c(3, 4, 5, 6, 7, 8)


## This long for loop repeats the experiment for 30 times, one for each simulated dataset

for(interactions in Number_of_Interactions){
  
  # Create all lists where to store results during the loop
  Resulting_AUC_DS <- list()
  Resulting_precision_DS <- list()
  Interactions_used <- list()
  Interactions_to_find <- interactions
  Tot_Variables_used <- list()
  Primary_Effect_Terms <- list()
  Secundary_Effect_Terms <- list()
  All_Effect_Terms <- list()
  Dataset <- list()
  Fitting_time <- list()
  
  for(n_sim in c(0:n_simulations)){
    
    # Get the right files. Files are generated through our custom python function and saved in a folder
    # where they
    filename = paste("hiprs_simdata_", n_sim, ".csv", sep = '')
    
    # Import and split Data 
    ds <- read_csv(filename)
    
    ds_train <- ds[c(1:1000),-1]
    ds_test <- ds[c(1001:1500),-1]
    
    y_ds_train = ds_train[,'Outcome']
    y_ds_train = y_ds_train[['Outcome']]
    
    y_ds_test = ds_test[,'Outcome']
    y_ds_test = y_ds_test[['Outcome']]
    
    ds_train <- ds_train[, c(1:ncol(ds_train)-1)]
    
    ds_test <- ds_test[, c(1:ncol(ds_test)-1)]
    
    ds_test <- lapply(ds_test, as.factor)
    ds_test <- as.data.frame(ds_test)
    
    ds_train <- lapply(ds_train, as.factor)
    ds_train <- as.data.frame(ds_train)
    
    
    levels <- list()
    
    ### Prepare parameters/objects needed to run glinternet function ###
    
    # For the simulated data we are using only the levels available within training set.
    # This is enough to run our simulated experiments as the levels in the training correspond to the levels
    # in the test set. This might not be the case for different datasets with various levels.
    # Change this manually by uncommenting lines if we want to consider the union of the levels available 
    # in the two datasets (training and test set) 
    
    for (i in c(1:ncol(ds_train))){
      levels_train <- list(levels(ds_train[,i]))
      #levels_test <- list(levels(ds_test[,i]))
      levels[i] <-levels_train
      #levels <- union(levels_train, levels_test) 
    }
    
    n_levels <- list()
    
    for (i in c(1:length(levels))){
      L <- length(unlist(levels[i]))
      n_levels[i] <- L
      
    }
    
    n_levels <- unlist(n_levels)
    
    n_features <- ncol(ds_train)
    numLevels <- n_levels
    
    ds_test <- lapply(ds_test, as.character)
    ds_test <- lapply(ds_test, as.numeric)
    ds_test <- as.data.frame(ds_test)
    
    ds_train <- lapply(ds_train, as.character)
    ds_train <- lapply(ds_train, as.numeric)
    ds_train <- as.data.frame(ds_train)
    #ds_train <- matrix(ds_train)
    
    
    ### GLINTERNET FIT ###
    
    start_time <- Sys.time()
    
    # FIT on Training Set
    fit_all = glinternet(ds_train, y_ds_train, numLevels, family = "binomial", numToFind = Interactions_to_find, nLambda = 100)
    
    # Save the fitting runtime
    end_time <- Sys.time()
    
    fitting_time <- end_time - start_time
    
    
    ### EXTRAPOLATE MODEL CHARACTERISTICS FROM THE FITTED GLINTERNET OBJECT ###
    
    # Get the last lambda, where the model stopped because it surpassed the Interactions_to_find threshold
    last_lambda = length(fit_all$activeSet)
    
    # Extrapolate how many interactions were actually used (note, it might be > Interactions_to_find)
    n_interactions = nrow(fit_all$activeSet[last_lambda][[1]]$catcat)
    
    coefs_result = coef(fit_all)
    
    n_main_effects = length(coefs_result[[last_lambda]]$mainEffectsCoef$cat)
    
    total_levels_primary_effect = 0
    
    for (i in c(1:n_main_effects)){
      
      total_levels_primary_effect <- total_levels_primary_effect + length(coefs_result[[last_lambda]]$mainEffectsCoef$cat[[i]])
      
    } 
    
    # Total number of variables applied, as sum of the unique primary effects and number of interactions 
    # (note: not considering levels, but only i.e. "A, B, AB" as single variables)
    tot_Vars_used = n_main_effects + n_interactions
    
    # Count how many levels were used for secundary effect terms, meaning the sum of the product of the
    # number of levels for each variable adopted in the interaction terms.
    total_levels_secundary_effect = length(fit_all$betahat[[last_lambda]])
    
    
    ### PREDICTION WITH THE FITTED GLINTERNET MODEL ###
    
    # PREDICT on Test Set
    predicted = as.vector(predict(fit_all, ds_test, "response", lambda = c(fit_all$lambda[last_lambda])))
    
    
    # Collect results and store them in lists
    
    roc_obj <- roc(y_ds_test, predicted)
    AUC <- auc(roc_obj)
    
    yy = as.factor(y_ds_test)
    
    AP <- average_precision_vec(yy, predicted)
    
    Resulting_precision_DS[position] <- AP
        
    Resulting_AUC_DS[position] <- AUC
    
    Interactions_used[position] <- n_interactions
    
    Tot_Variables_used[position] <- tot_Vars_used
    
    Primary_Effect_Terms[position] <- total_levels_primary_effect
    
    Secundary_Effect_Terms[position] <- total_levels_secundary_effect
    
    All_Effect_Terms[position] <- total_levels_primary_effect + total_levels_secundary_effect
    
    Fitting_time[position] <- fitting_time
    
    Dataset[position] <- n_sim
    
    position = position + 1
    
  }                                                                                                                                                                 
  
  # AVERAGE PRECISION OVERALL RESULTS
  Resulting_precision_DS <- unlist(Resulting_precision_DS)
  mean_prec <- mean(Resulting_precision_DS)
  sd_prec <- sd(Resulting_precision_DS)
  
  # AUC OVERALL RESULTS
  Resulting_AUC_DS <- unlist(Resulting_AUC_DS)
  mean_AUC <- mean(Resulting_AUC_DS)
  sd_AUC <- sd(Resulting_AUC_DS)
  
  Interactions_used <- unlist(Interactions_used)

  Tot_Variables_used <- unlist(Tot_Variables_used)

  Primary_Effect_Terms <- unlist(Primary_Effect_Terms)
  Secundary_Effect_Terms <- unlist(Secundary_Effect_Terms)
  All_Effect_Terms <- unlist(All_Effect_Terms)
  
  Fitting_time <- unlist(Fitting_time)
  
  Dataset <- unlist(Dataset)
  
  
  results_df = data.frame(Dataset, Resulting_AUC_DS, Resulting_precision_DS, Interactions_used, Tot_Variables_used, Primary_Effect_Terms, Secundary_Effect_Terms, All_Effect_Terms, Fitting_time)
  
  results_df$Proposed_N_interactions <- Interactions_to_find
  results_df$Mean_AUC <- mean_AUC
  results_df$SD_AUC <- sd_AUC
  results_df$Mean_prec <- mean_prec
  results_df$SD_prec <- sd_prec
  
  RESULTS <- rbind(RESULTS, results_df)
  
}



write.csv(RESULTS, "GLINTERNET_sim_results_NEW.csv")




########### GLINTERNET INTERPRETABILITY EXPERIMENT #########

## The following code repeats glinternet fitting only on the dataset 
## used in the paper to compare methods in terms of interpretability.
## It selects the proper dataset, fits glinternet and extracts 
## model parameters for inspection.

n_sim = 5

Interactions_to_find = 3


# Get the right files
filename = paste("hiprs_simdata_", n_sim, ".csv", sep = '')

# Import and Preprocess Data 
ds <- read_csv(filename)


ds_train <- ds[c(1:1000),-1]
ds_test <- ds[c(1001:1500),-1]

y_ds_train = ds_train[,'Outcome']
y_ds_train = y_ds_train[['Outcome']]

y_ds_test = ds_test[,'Outcome']
y_ds_test = y_ds_test[['Outcome']]

ds_train <- ds_train[, c(1:ncol(ds_train)-1)]

ds_test <- ds_test[, c(1:ncol(ds_test)-1)]

ds_test <- lapply(ds_test, as.factor)
ds_test <- as.data.frame(ds_test)

ds_train <- lapply(ds_train, as.factor)
ds_train <- as.data.frame(ds_train)


levels <- list()

for (i in c(1:ncol(ds_train))){
  levels_train <- list(levels(ds_train[,i]))
  levels[i] <-levels_train
}

n_levels <- list()

for (i in c(1:length(levels))){
  L <- length(unlist(levels[i]))
  n_levels[i] <- L
  
}

n_levels <- unlist(n_levels)

n_features <- ncol(ds_train)
numLevels <- n_levels

ds_test <- lapply(ds_test, as.character)
ds_test <- lapply(ds_test, as.numeric)
ds_test <- as.data.frame(ds_test)

ds_train <- lapply(ds_train, as.character)
ds_train <- lapply(ds_train, as.numeric)
ds_train <- as.data.frame(ds_train)
#ds_train <- matrix(ds_train)


## Start the actual GLINTERNET application


# FIT on Training Set
fit_all = glinternet(ds_train, y_ds_train, numLevels, family = "binomial", numToFind = Interactions_to_find, nLambda = 100)


# Get the last lambda, where the model stopped because it surpassed the Interactions_to_find threshold
last_lambda = length(fit_all$activeSet)

# Extrapolate how many interactions were actually used (note, it might be > Interactions_to_find)
fit_all$activeSet[last_lambda][[1]]$catcat

#get the coefficients for the model
coefs_result = coef(fit_all)

coefs_result[[last_lambda]]$mainEffects
#  $cat
#  [1] 7 8 9

#get the main effects coefficients
main_effects = data.frame(c(coefs_result[[last_lambda]]$mainEffectsCoef[[1]][[1]], coefs_result[[last_lambda]]$mainEffectsCoef[[1]][[2]], coefs_result[[last_lambda]]$mainEffectsCoef[[1]][[3]]))


#save the model's coefficients for the interactions
df1 = data.frame(coefs_result[[last_lambda]]$interactionsCoef$catcat[[1]])
df2 = data.frame(coefs_result[[last_lambda]]$interactionsCoef$catcat[[2]])
df3 = data.frame(coefs_result[[last_lambda]]$interactionsCoef$catcat[[3]])

interactions_coeff = cbind(df1, df2)
df3

write.csv(interactions_coeff, "GLINTERNET_sim_results_parametersInt_par1_5.csv")

write.csv(df3, "GLINTERNET_sim_results_parametersInt_part2_5.csv")

write.csv(main_effects, "GLINTERNET_sim_results_parametersMainEffects_5.csv")






