# Ayesha Code --------------------------------
library(data.table)
library(rpart)
library(rpart.plot)   
library(ggplot2)
library(caTools)
library(caret)

setwd("C:/Users/JianBo/OneDrive - Nanyang Technological University/Y3S1/BC2406 Analytics 1/Project")
dt <- fread("predictive_maintenance.csv")
summary(dt)

# rename columns 
colnames(dt) <- c('UDI', 'ID', 'type', 'air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear', 'failure', 'failure_type')

# changing 0/1 to no failure/failure 
dt[, failure := factor(failure, levels = 0:1, labels = c("no failure", "failure"))]
levels(dt$failure)

set.seed(1)

# take out UDI, ID (irrelevant) and failure_type (target)
tree <- rpart(failure ~ . -failure_type-UDI-ID , data = dt, method = 'class', control = rpart.control(cp = 0)) # find maximal tree first 
rpart.plot(tree, nn = T, main = "Maximal Tree")
print(tree)
printcp(tree)

CVerror <- tree$cptable[which.min(tree$cptable[,"xerror"]), "xerror"] # find min xerror 
maxCVerror <- CVerror + tree$cptable[which.min(tree$cptable[,"xerror"]), "xstd"] # calculate cv error cap 


# finding optimal tree 
i <- 1; j <- 4
while(tree$cptable[i,j] > maxCVerror){
  i <- i+1
}

cp <- ifelse(i > 1, sqrt(tree$cptable[i,1] * tree$cptable[i-1,1]), 1)
pruned_tree <- prune(tree, cp = cp)
rpart.plot(pruned_tree)
printcp(pruned_tree)
print(pruned_tree)


## USING TRAIN-TEST SPLIT TO MAKE THE MODEL


sample_train <- sample.split(Y = dt$failure, SplitRatio = 0.70)
trainset <- subset(x = dt, sample_train == T)
testset <- subset(x = dt, sample_train == F)

trainset[, .N, failure] # data is unbalanced, 6763 no failure, 237 failure 

# sample the majority to address imbalanced data, but use same testset for testing 
majority <- trainset[failure == "no failure"]
minority <- trainset[failure == "failure"]

# randomly sample the row numbers to be in trainset. sample size can be up to 2x minority cases. choose 1.5x
chosen <- sample(seq(1:nrow(majority)), size = 1.5*(nrow(minority)))
majority.chosen <- majority[chosen]

# combine two data tables by appending the rows
trainset.bal <- rbind(majority.chosen, minority)
summary(trainset.bal)

# check trainset is balanced
trainset.bal[, .N, failure] # 355 no failure, 237 failure 


# finding optimal tree using balanced trainset 
traintree <- rpart(failure ~ . -failure_type-UDI-ID , data = trainset.bal, method = 'class', control = rpart.control(minsplit = 2, cp = 0)) # find maximal tree first 
rpart.plot(traintree, nn = T, main = "Maximal Tree")
print(traintree)
printcp(traintree)
plotcp(traintree)

CVerror <- traintree$cptable[which.min(traintree$cptable[,"xerror"]), "xerror"]
maxCVerror <- CVerror + traintree$cptable[which.min(traintree$cptable[,"xerror"]), "xstd"]

i <- 1; j <- 4
while(traintree$cptable[i,j] > maxCVerror){
  i <- i+1
}

cp <- ifelse(i > 1, sqrt(traintree$cptable[i,1] * traintree$cptable[i-1,1]), 1)
pruned_tree <- prune(traintree, cp = cp)
rpart.plot(pruned_tree, nn=T, extra = 101, main = "Optimal Tree for predicting machine failure")
print(pruned_tree)
summary(pruned_tree) 


## VARIABLE SIGNIFICANCE ##
pruned_tree$variable.importance 
sum = sum(pruned_tree$variable.importance)
round((pruned_tree$variable.importance/sum)*100)


## EVALUATING THE MODEL ##


# now, we use the testset as input into our optimal tree
prediction <- predict(pruned_tree, testset, type="class")
prediction
mean(prediction == testset$failure)

# to evaluate the effectiveness of our model for a categorical Y, we use a confusion matrix 
CM <- confusionMatrix(prediction,testset$failure) 
CM

table <- table(Actual_Values = testset$failure, Prediction = prediction, deparse.level = 2)
table
round(prop.table(table),3)


# training on unbalanced data: accuracy 97%, misclassification error 3%
# training on balanced data: accuracy 88.77%, misclassification error 11.23%, accuracy decreases. 

# CHRIS CODE -----------------------------------
install.packages("car")
install.packages("caret")
install.packages("carData")
library(ggplot2)
library(caret)
library(car)
library(data.table)

setwd("C:/Users/JianBo/OneDrive - Nanyang Technological University/Y3S1/BC2406 Analytics 1/Project")
dt <- fread("predictive_maintenance.csv")
summary(dt)

# rename columns 
colnames(dt) <- c('UDI', 'ID', 'type', 'air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear', 'failure', 'failure_type')
summary(dt)

# changing 0/1 to no failure/failure 
dt[, failure := factor(failure, levels = 0:1, labels = c("no failure", "failure"))]
levels(dt$failure)

set.seed(1)
summary(dt)

# take out UDI, ID (irrelevant) and failure_type (target)
dt <- dt[!is.na(dt$failure), ]

library(dplyr)

dt <- dt %>%
  mutate(across(.cols = all_of(c("failure_type", "UDI", "ID")), 
                as.factor)) %>%
  select(-failure_type, -UDI, -ID)

summary(dt)
## USING TRAIN-TEST SPLIT TO MAKE THE MODEL

# Set a random seed for reproducibility
set.seed(1)

# Generate a random sample of row indices for the training set
sample_indices <- sample(nrow(dt), 0.7 * nrow(dt))

# Create the training set and testing set
trainset <- dt[sample_indices, ]
testset <- dt[-sample_indices, ]

trainset[, .N, failure] # data is unbalanced, 6763 no failure, 237 failure 

# sample the majority to address imbalanced data, but use same testset for testing 
majority <- trainset[failure == "no failure"]
minority <- trainset[failure == "failure"]

# randomly sample the row numbers to be in trainset. sample size can be up to 2x minority cases. choose 1.5x
chosen <- sample(seq(1:nrow(majority)), size = 1.5*(nrow(minority)))
majority.chosen <- majority[chosen]

# combine two data tables by appending the rows
trainset.bal <- rbind(majority.chosen, minority)
summary(trainset.bal)

# check trainset is balanced
trainset.bal[, .N, failure] # 355 no failure, 237 failure 

full_model <- glm(failure ~ ., data = trainset.bal, family = binomial(link = "logit"))
# Perform backward elimination
reduced_model <- step(full_model, direction = "backward")

vif_values <- vif(reduced_model)

summary(reduced_model)

prediction <- predict(reduced_model, newdata = testset, type = "response")
prediction <- ifelse(prediction > 0.5, "failure", "no failure") # Convert probabilities to binary predictions

# Calculate accuracy
accuracy <- mean(prediction == testset$failure)
accuracy
# Create a confusion matrix
levels(testset$failure)
prediction <- factor(prediction, levels = levels(testset$failure))
confusion_matrix <- confusionMatrix(prediction, testset$failure)
confusion_matrix

# Sailesh Code ----------------------------------------
library(data.table)
library(rpart)
library(rpart.plot)   
library(ggplot2)
library(caTools)
library(caret)
library(car)
library(e1071)
dt1 <- fread("global-data-on-sustainable-energy.csv")
summary(dt1)
# View(dt1)
colnames(dt1) <- c('Country', 'Year', 'ElectricityAccess', 'CleanFuelAccess', 'RenewableElectricityCap', 'FinancialFlows', 
                   'RenewableEnergyShare', 'FossilFuelElectricity', 'NuclearElectricity', 'RenewableElectricity', 
                   'LowCarbonElectricityShare', 'EnergyConsumptionPerCapita', 'EnergyIntensity', 'CO2Emissions', 
                   'RenewableEnergyPercent', 'GDPGrowth', 'GDPPerCapita', 'PopulationDensity', 'LandArea', 'Latitude', 'Longitude')
set.seed(1)
dt1 <- as.data.table(dt1)
dt <- dt1[dt1$Year == 2019]
summary(dt)
na_rows_column <- which(is.na(dt$EnergyIntensity))
na_rows_column
na_rows_column2 <- which(is.na(dt$CO2Emissions))
na_rows_column2
na_rows_column3 <- which(is.na(dt$GDPGrowth))
na_rows_column3
na_rows_column4 <- which(is.na(dt$RenewableEnergyShare))
na_rows_column4

dt <- dt[is.na(dt$RenewableEnergyShare)==F,]
dt$CO2Emissions <- ifelse(is.na(dt$CO2Emissions), median(dt$CO2Emissions, na.rm = TRUE), dt$CO2Emissions)
summary(dt$CO2Emissions)
dt$EnergyIntensity <- ifelse(is.na(dt$EnergyIntensity), median(dt$EnergyIntensity, na.rm = TRUE), dt$EnergyIntensity)
summary(dt$EnergyIntensity)
dt$GDPGrowth <- ifelse(is.na(dt$GDPGrowth), mean(dt$GDPGrowth, na.rm = TRUE), dt$GDPGrowth)
summary(dt$GDPGrowth)

dt$EnergyIntensity_log <- log(dt$EnergyIntensity)
dt$CO2Emissions_log <- log(dt$CO2Emissions)
train <- sample.split(dt$Country, SplitRatio = 0.7)
train
trainset <- subset(dt, train == T)
trainset
testset <- subset(dt, train ==F)
testset

m1  <- lm(RenewableEnergyShare ~ EnergyIntensity_log + CO2Emissions_log + GDPGrowth, data = trainset)
summary(m1)
vif(m1)

par(mfrow = c(2,2))
plot(m1)
residuals(m1)

m2 <- step(m1)
summary(m2)
vif(m2)
plot(m2)
residuals(m2)
# Residuals = Error = Actual - Model Predicted
residuals_train <- residuals(m1)

# RMSE on trainset
RMSE.m1.train <- sqrt(mean(residuals_train^2))
RMSE.m1.train

# Check Min and Max Absolute Error on Training Set
summary(abs(residuals_train))

# Apply model from trainset to predict on testset
predictions_test <- predict(m1, newdata = testset)

# Calculate residuals on test set
residuals_test <- testset$RenewableEnergyShare - predictions_test

# RMSE on testset
RMSE.m1.test <- sqrt(mean(residuals_test^2, na.rm = TRUE))
RMSE.m1.test
# Check Min and Max Absolute Error on Test Set
summary(abs(residuals_test))

# Residuals = Error = Actual - Model Predicted
residuals_train2 <- residuals(m2)

# RMSE on trainset
RMSE.m2.train <- sqrt(mean(residuals_train^2))
RMSE.m2.train

# Check Min and Max Absolute Error on Training Set
summary(abs(residuals_train2))

# Apply model from trainset to predict on testset
predictions_test2 <- predict(m2, newdata = testset)

# Calculate residuals on test set
residuals_test2 <- testset$RenewableEnergyShare - predictions_test2

# RMSE on testset
RMSE.m2.test <- sqrt(mean(residuals_test2^2, na.rm = TRUE))
RMSE.m2.test
# Check Min and Max Absolute Error on Test Set
summary(abs(residuals_test2))

# Jian Bo Code-----------------------------------------------------
library(data.table)
library(caTools)
library(rpart)
library(caret)
library(rpart.plot)
library(ggplot2)
library(corrplot)
library(e1071)


setwd("C:/Users/JianBo/OneDrive - Nanyang Technological University/Y3S1/BC2406 Analytics 1/Project")
dt1 <- fread("global-data-on-sustainable-energy.csv")
levels(dt1$Entity)
set.seed(1)

colnames(dt1) <- c('Country', 'Year', 'ElectricityAccess', 'CleanFuelAccess', 'RenewableElectricityCap', 'FinancialFlows', 
                   'RenewableEnergyShare', 'FossilFuelElectricity', 'NuclearElectricity', 'RenewableElectricity', 
                   'LowCarbonElectricityShare', 'EnergyConsumptionPerCapita', 'EnergyIntensity', 'CO2Emissions', 
                   'RenewableEnergyPercent', 'GDPGrowth', 'GDPPerCapita', 'PopulationDensity', 'LandArea', 'Latitude', 'Longitude')
par(mfrow=c(2,2))
dt1
dt <- dt1[dt1$Year == 2019,]
dt <- dt[is.na(RenewableEnergyShare)==F,]
dt$CO2Emissions <- as.numeric(dt$CO2Emissions)
dt$PopulationDensity <- as.numeric(dt$PopulationDensity)
summary(dt$CO2Emissions)
summary(dt$EnergyIntensity)
skewness(dt$RenewableEnergyShare)
skewness(dt$EnergyIntensity, na.rm = TRUE)
skewness(dt$CO2Emissions, na.rm = TRUE)
skewness(dt$GDPGrowth, na.rm = TRUE)
plot(density(dt$EnergyIntensity, na.rm = TRUE))
plot(density(dt$RenewableEnergyShare, na.rm = TRUE))
plot(density(dt$CO2Emissions, na.rm = TRUE))
na_rows_column <- which(is.na(dt$EnergyIntensity))
na_rows_column
na_rows_column2 <- which(is.na(dt$CO2Emissions))
na_rows_column2
na_rows_column3 <- which(is.na(dt$GDPGrowth))
na_rows_column3
na_rows_column4 <- which(is.na(dt$RenewableEnergyShare))
na_rows_column4

dt$CO2Emissions <- ifelse(is.na(dt$CO2Emissions), median(dt$CO2Emissions, na.rm = TRUE), dt$CO2Emissions)
summary(dt$CO2Emissions)
dt$EnergyIntensity <- ifelse(is.na(dt$EnergyIntensity), median(dt$EnergyIntensity, na.rm = TRUE), dt$EnergyIntensity)
summary(dt$EnergyIntensity)
dt$GDPGrowth <- ifelse(is.na(dt$GDPGrowth), mean(dt$GDPGrowth, na.rm = TRUE), dt$GDPGrowth)
summary(dt$GDPGrowth)

skewness(dt$RenewableEnergyShare)
skewness(dt$EnergyIntensity)
skewness(dt$CO2Emissions)
skewness(dt$GDPGrowth)
dt$EnergyIntensity_log <- log(dt$EnergyIntensity)
dt$CO2Emissions_log <- log(dt$CO2Emissions)
summary(dt$EnergyIntensity)
summary(dt$CO2Emissions)
summary(dt$GDPGrowth)
train <- sample.split(dt$Country, SplitRatio = 0.7)
train
trainset <- subset(dt, train == T)
trainset
testset <- subset(dt, train ==F)
testset
cart1 <- rpart(RenewableEnergyShare ~ CO2Emissions_log + EnergyIntensity_log + GDPGrowth, data = trainset, method = 'anova', control = rpart.control(minsplit = 2, cp = 0))
rpart.plot(cart1, nn = T, main ="Maximal CART")
print(cart1)
printcp(cart1)
plotcp(cart1)

# finding optimal tree
#store the cptable
cp <- data.table(cart1$cptable)
#number the sequence of the tree_us
cp[,index:=1:nrow(cp)]
cp
#find out minimum index where xerror is min
min_cp_index <- min(cp[(xerror+xstd==min(xerror+xstd)),index])
min_cp_index
#find the errorcap
errorcap <- cp[min_cp_index, xerror + xstd]
errorcap
#find our the optimal index for the cp
optimal_cp_index <- min(cp[(xerror<errorcap),index])
optimal_cp_index
cp.opt <- sqrt(cp[index==optimal_cp_index, CP]*cp[index==optimal_cp_index-1, CP])
#you are done - trained optimal model
cp.opt

m.opt <- prune(cart1, cp = cp.opt)
m.opt
par(mfrow = c(1,1))
rpart.plot(m.opt, nn= T, extra = 101,main = "Optimal Tree of renewable energy consumption %")
summary(cart1)
summary(m.opt)
?rpart.plot
# rpart.plot(m.opt, cex = 0.9)
plotcp(m.opt)

# variable importance
m.opt$variable.importance
scaledVarImpt_us <- round(100*m.opt$variable.importance/sum(m.opt$variable.importance))
scaledVarImpt_us[scaledVarImpt_us > 3]

# rmse trainset
rmse.train <- sqrt(mean(residuals(m.opt)^2))
rmse.train

# rmse testset
cart1.predict <- predict(m.opt,newdata = testset)
cart1.predict
rmse.test <- sqrt(mean((testset$RenewableEnergyShare - cart1.predict)^2))
rmse.test

rmse.train
rmse.test

