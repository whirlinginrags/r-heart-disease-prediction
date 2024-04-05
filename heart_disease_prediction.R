# uncomment the line below if any of the packages are not installed
# install.packages(c("tidyverse", "caret", "e1071", "randomForest", "kernlab", "xgboost", "pROC", "labeling"))

# libraries
library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(kernlab)
library(xgboost)
library(pROC)
library(labeling)  
library(scales)

# 1. data should be visualized by analyzing the data.
# loading dataset
data <- read.csv("heart.csv")

# visualize data
summary(data)
ggplot(data, aes(x = factor(target), fill = factor(target))) + geom_bar()

# 2. in the pre-processing stage, missing data should be filled with the average value of that column.
# filling missing data with mean
data <- data %>% mutate_all(list(~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# 3. normalization should be done
numeric_columns <- sapply(data, is.numeric)

data[, numeric_columns] <- apply(data[, numeric_columns], 2, function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))

# 4. find outlier data and write a function to replace the outlier data with the maximum non-outlier value in that column.
replace_outliers <- function(x) {
  q <- quantile(x, c(0.25, 0.75))
  iqr <- IQR(x)
  lower_bound <- q[1] - 1.5 * iqr
  upper_bound <- q[2] + 1.5 * iqr
  x[x < lower_bound] <- max(x)
  x[x > upper_bound] <- max(x)
  return(x)
}

# apply the function to columns with numeric data
data[, numeric_columns] <- apply(data[, numeric_columns], 2, replace_outliers)

#  5. perform at least 1 new feature extraction
# creating a new feature by combining two existing features
data$new_feature <- data$oldpeak * data$thalach

data$target <- as.factor(data$target)

# 6. classify the data with at least 5 classifiers
# split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$target, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# train models
model1 <- train(target ~ ., data = train_data, method = "glm", trControl = trainControl(method = "cv"))
model2 <- train(target ~ ., data = train_data, method = "knn", trControl = trainControl(method = "cv"))
model3 <- train(target ~ ., data = train_data, method = "rf", trControl = trainControl(method = "cv"))
model4 <- train(target ~ ., data = train_data, method = "svmRadial", trControl = trainControl(method = "cv"))
model5 <- train(target ~ ., data = train_data, method = "xgbTree", trControl = trainControl(method = "cv"),
                tuneGrid = expand.grid(nrounds = 100, max_depth = 3, eta = 0.1, gamma = 0, colsample_bytree = 1, min_child_weight = 1, subsample = 1),
                metric = "Accuracy")

# predictions
pred1 <- predict(model1, newdata = test_data)
pred2 <- predict(model2, newdata = test_data)
pred3 <- predict(model3, newdata = test_data)
pred4 <- predict(model4, newdata = test_data)
pred5 <- predict(model5, newdata = test_data)

# confusion matrices
conf_matrix1 <- confusionMatrix(pred1, test_data$target)
conf_matrix2 <- confusionMatrix(pred2, test_data$target)
conf_matrix3 <- confusionMatrix(pred3, test_data$target)
conf_matrix4 <- confusionMatrix(pred4, test_data$target)
conf_matrix5 <- confusionMatrix(pred5, test_data$target)

# 7. show the accuracy, sensitivity, specificity, F1-Score values of the results of the classifiers
get_metrics <- function(conf_matrix) {
  accuracy <- sum(diag(as.matrix(conf_matrix$table))) / sum(as.matrix(conf_matrix$table))
  sensitivity <- conf_matrix$table[2, 2] / sum(conf_matrix$table[2, ])
  specificity <- conf_matrix$table[1, 1] / sum(conf_matrix$table[1, ])
  f1_score <- 2 * (sensitivity * specificity) / (sensitivity + specificity)
  
  return(c(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity, F1_Score = f1_score))
}

# 8. show the confusion matrix for each classifier
print("model 1: logistic regression")
print(conf_matrix1)
cat("additional metrics:")
print(get_metrics(conf_matrix1))

print("model 2: k-nn")
print(conf_matrix2)
cat("additional metrics:")
print(get_metrics(conf_matrix2))

print("model 3: random forest")
print(conf_matrix3)
cat("additional metrics:")
print(get_metrics(conf_matrix3))

print("model 4: svm radial")
print(conf_matrix4)
cat("additional metrics:")
print(get_metrics(conf_matrix4))

print("model 5: xgboost")
print(conf_matrix5)
cat("additional metrics:")
print(get_metrics(conf_matrix5))

# 9. visualize the classifier results
roc_curve1 <- roc(test_data$target, as.numeric(pred1))
roc_curve2 <- roc(test_data$target, as.numeric(pred2))
roc_curve3 <- roc(test_data$target, as.numeric(pred3))
roc_curve4 <- roc(test_data$target, as.numeric(pred4))
roc_curve5 <- roc(test_data$target, as.numeric(pred5))
roc_curve_best <- roc(test_data$target, as.numeric(pred5))

plot(roc_curve1, col = "blue", main = "receiver operating characteristic (roc)")
lines(roc_curve2, col = "red")
lines(roc_curve3, col = "green")
lines(roc_curve4, col = "purple")
lines(roc_curve5, col = "orange")

# legend dynamically indicating the best model
legend("bottomright", legend = c(
  paste("model 1: logistic regression", ifelse(1 == which.max(accuracies), "(best)", "")),
  paste("model 2: k-nn", ifelse(2 == which.max(accuracies), "(best)", "")),
  paste("model 3: random forest", ifelse(3 == which.max(accuracies), "(best)", "")),
  paste("model 4: svm radial", ifelse(4 == which.max(accuracies), "(best)", "")),
  paste("model 5: xgboost", ifelse(5 == which.max(accuracies), "(best)", ""))
), col = c("blue", "red", "green", "purple", "orange"), lty = c(1, 1, 1, 1, 1))

# 10. save the model with the best accuracy.
accuracies <- sapply(list(model1, model2, model3, model4, model5), function(model) max(model$results$Accuracy))
best_model_index <- which.max(accuracies)
best_model <- list(model1, model2, model3, model4, model5)[best_model_index]
saveRDS(best_model, "best_model.rds")

# print the best model name
best_model_name <- c("logistic regression", "k-nn", "random forest", "svm radial", "xgboost")[best_model_index]
cat("best performing model:", best_model_name, "\n")

# 11. when new data comes in, predict according to the best model.
new_data <- read.csv("newdata.csv") # change the newdata.csv to your data file for predictions. for testing purposes i have removed the target column from the dataset you have provided and predicted the results once again.
new_data <- new_data %>% mutate_all(list(~ifelse(is.na(.), mean(., na.rm = TRUE), .)))
new_data[, numeric_columns] <- apply(new_data[, numeric_columns], 2, function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
new_data[, numeric_columns] <- apply(new_data[, numeric_columns], 2, replace_outliers)
new_data$new_feature <- new_data$oldpeak * new_data$thalach

prediction <- predict(best_model, newdata = new_data)
cat("prediction with the best performing model:")
print(prediction)
