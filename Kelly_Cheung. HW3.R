##################################################
# ECON 418-518 Homework 3
# Kelly Cheung
# The University of Arizona
# waiyin03@arizona.edu 
# 7 Dec 2024
###################################################


#####################
# Preliminaries
#####################

# Clear environment, console, and plot pane
rm(list = ls())
cat("\014")
graphics.off()

# Turn off scientific notation
options(scipen = 999)

# Load packages
pacman::p_load(data.table)

# Set sead
set.seed(418518)


#####################
# Problem 1
#####################

#################
# Question (i)
#################

# Set the working directory to the folder
setwd("/Users/kellycheung/Downloads")

# Verify the working directory
getwd()

# Load the CSV file
data <- read.csv("ECON_418-518_HW3_Data.csv")
library(data.table)

# Load the data
dt <- fread("ECON_418-518_HW3_Data.csv")

# Drop specified columns
dt <- dt[, !c("fnlwgt", "occupation", "relationship", "capital-gain", "capital-loss", "educational-num"), with = FALSE]

##############
# Question (ii)
##############

 (a) # Convert 'income' to a binary indicator
dt$income <- ifelse(dt$income == ">50K", 1, 0)

(b) # Convert 'race' to binary indicator (White = 1, otherwise 0)
dt$race <- ifelse(dt$race == "White", 1, 0)

(c) # Convert 'gender' to binary indicator (Male = 1, otherwise 0)
dt$gender <- ifelse(dt$gender == "Male", 1, 0)

(d) # Convert 'workclass' to binary indicator (Private = 1, otherwise 0)
dt$workclass <- ifelse(dt$workclass == "Private", 1, 0)

(e) # Convert 'native country' to binary indicator (United-States = 1, otherwise 0)
dt$native.country <- ifelse(dt$native.country == "United-States", 1, 0)
dt$native.country
(f) # Convert 'marital status' to binary indicator (Married-civ-spouse = 1, otherwise 0)
dt$marital.status <- ifelse(dt$marital.status == "Married-civ-spouse", 1, 0)
dt$marital.status

(g) # Convert 'education' to binary indicator (Bachelors, Masters, Doctorate = 1, otherwise 0)
dt$education <- ifelse(dt$education %in% c("Bachelors", "Masters", "Doctorate"), 1, 0)

(h) # Create 'age sq' as the squared value of age
dt$age_sq <- dt$age^2

(i) # Standardize 'age', 'age_sq', and 'hours per week'
dt$age <- scale(dt$age)
dt$age_sq <- scale(dt$age_sq)
dt$`hours-per-week`<- scale(dt$`hours-per-week`)

##################
# Question (iii)
##################

#(a) Proportion of individuals with income > 50K
prop_income_50k <- mean(dt$income == 1)
prop_income_50k 

# (b) Proportion of individuals in the private sector
prop_private_sector <- mean(dt$workclass == 1)
prop_private_sector

#(c) Proportion of married individuals
prop_married <- mean(dt$marital.status == 1)
prop_married 

 #(d) Proportion of females
prop_females <- mean(dt$gender == 0)
prop_females

# (e)Count of missing values (NAs)
total_na <- sum(is.na(dt))
total_na 

# (f)Convert 'income' to factor
dt$income <- factor(dt$income)

################

# Question (iv)

################

# Set the seed for reproducibility
set.seed(418518)

# Calculate the number of training samples (70% of total)
train_size <- floor(nrow(dt) * 0.70)

# Split the data into training and testing sets
train_data <- dt[1:train_size, ]
test_data <- dt[(train_size + 1):nrow(dt), ]


##############
# Question (v)
##############

install.packages("caret")
install.packages("glmnet")
install.packages("e1071")
library(caret)
library(glmnet)

#Check which columns have missing values and their counts
colSums(is.na(train_data))

#Remove Rows with Missing Values
train_data <- na.omit(train_data)

# Replace NAs with column mean
train_data[is.na(train_data)] <- lapply(train_data, function(x) 
  ifelse(is.numeric(x), mean(x, na.rm = TRUE), x))

preProc <- preProcess(train_data, method = "medianImpute")
train_data <- predict(preProc, train_data)

# Impute missing numeric columns with their median
for (col in names(train_data)) {
  if (is.numeric(train_data[[col]])) {
    train_data[[col]][is.na(train_data[[col]])] <- median(train_data[[col]], na.rm = TRUE)
  }
}

# Impute missing factor columns with the most frequent value
for (col in names(train_data)) {
  if (is.factor(train_data[[col]])) {
    levels <- table(train_data[[col]])
    most_frequent <- names(which.max(levels))
    train_data[[col]][is.na(train_data[[col]])] <- most_frequent
  }
}


ctrl <- trainControl(method = "cv", number = 10)
train_data[apply(is.na(train_data), 1, any), ]  # View rows with missing values

####################
missing_rows <- train_data[apply(is.na(train_data), 1, any), ]
print(missing_rows)  # This will show rows with missing values
train_data <- train_data[complete.cases(train_data), ]
for (col in names(train_data)) {
  if (is.numeric(train_data[[col]])) {
    train_data[[col]][is.na(train_data[[col]])] <- median(train_data[[col]], na.rm = TRUE)
  }
}
for (col in names(train_data)) {
  if (is.factor(train_data[[col]])) {
    most_frequent <- names(which.max(table(train_data[[col]], useNA = "no")))
    train_data[[col]][is.na(train_data[[col]])] <- most_frequent
  }
}
sum(is.na(train_data))  # Should return 0
ctrl <- trainControl(method = "cv", number = 10)

lasso_model <- train(income ~ ., data = train_data, method = "glmnet", 
                     trControl = ctrl, tuneLength = 50)

# Best lambda value
best_lambda_lasso <- lasso_model$bestTune$lambda

# Classification accuracy
lasso_accuracy <- lasso_model$results$Accuracy[lasso_model$results$lambda == best_lambda_lasso]
str(train_data)
summary(train_data)
nrow(train_data)
lasso_coeffs <- coef(lasso_model$finalModel, s = best_lambda_lasso)
lasso_coeffs


#############
# Question (vi)
#############
install.packages(randomForest)
install.packages(caret)
library(randomForest)
library(caret)

# Train random forest models
rf_model_100 <- randomForest(income ~ ., data = train_data, ntree = 100, mtry = 2)
rf_model_200 <- randomForest(income ~ ., data = train_data, ntree = 200, mtry = 5)
rf_model_300 <- randomForest(income ~ ., data = train_data, ntree = 300, mtry = 9)

# Evaluate accuracy for each model
accuracy_100 <- mean(predict(rf_model_100, test_data) == test_data$income)
accuracy_200 <- mean(predict(rf_model_200, test_data) == test_data$income)
accuracy_300 <- mean(predict(rf_model_300, test_data) == test_data$income)


################
# Question (vii)
################

# Best model based on accuracy (assuming rf_model_300 is best)
best_rf_model <- rf_model_300
best_rf_model
# Evaluate classification accuracy on test data
best_rf_accuracy <- mean(predict(best_rf_model, test_data) == test_data$income)
best_rf_accuracy 
# Create confusion matrix
confusion_matrix <- confusionMatrix(predict(best_rf_model, test_data), test_data$income)
confusion_matrix

# Use the best model to predict on the testing set
predictions <- predict(best_model, newdata = test_data)

# Calculate classification accuracy
accuracy <- mean(predictions == test_data$income)
print(accuracy)




















