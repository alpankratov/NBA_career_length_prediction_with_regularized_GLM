library(tidyverse)
library(ggplot2)
library(ggthemes)
library(GGally)
library(glmnet)
library(ggcorrplot)
library(gridExtra)
library(caret)
library(plotROC)
library(e1071)


# PREPARATION OF DATA---------------------------------
# loading the data
data <- read_csv('data/data.csv') %>% dplyr::select(-X1) %>% # the first variable is removed as it is just a row number
  mutate(Target = factor(Target, ordered = FALSE, labels = c("F", "S")),
         Year_drafted = factor(Year_drafted, ordered = TRUE))
# where F - is failed to have career more than 5 seasons and S is suceeded in it

# splitting data between training (70%) and test (30%)
sample_size <- floor(0.7 * nrow(data))
set.seed(321)
training_index <- sample(seq_len(nrow(data)), size = sample_size)

# Creating data set for exploratory analysis that contains the same observations as the training dataset but includes
# all variables that are not centered and scaled
data_explore <- data[training_index, ]

# Remove veriables Yrs and Name as Target is directly derived from Yrs and Name has unique value for each observation
data_train <- data[training_index, ] %>% dplyr::select(-c('Name','Yrs'))
data_test <- data[-training_index, ] %>% dplyr::select(-c('Name','Yrs'))

# We should nowstandardise all variables (besides the response variable) using the mean and variance fitted
# on the training dataset to avoid data leakage.
scalingData <- preProcess(data_train[,1:20], method = c("center", "scale"))
data_train[,1:20] <- predict(scalingData,data_train[,1:20])
data_test[,1:20] <- predict(scalingData,data_test[,1:20])

# EXPLORATORY ANALYSIS -----------------------------------------------------

# Number of observations
nrow(data)
nrow(data_test)
nrow(data_train)

# Career length distribution
ggplot(data_explore, aes(x=Yrs)) + geom_bar(fill = '#4BAEFA') +
  geom_vline(xintercept = mean(as_vector(data_explore['Yrs'])), color = '#FF9A3D', size = 1.5) +
  annotate('text', x = 13, y = 55,
           label = paste("Mean career length - ", round(mean(as_vector(data_explore['Yrs'])), 1), " years (orange line)")) +
  geom_vline(xintercept = median(as_vector(data_explore['Yrs'])), color = '#FFE017', size = 1.5) +
  annotate('text', x = 13, y = 45,
           label = paste("Median career length - ", round(median(as_vector(data_explore['Yrs'])), 1), " years (yellow line)")) +
  xlab('Career length (in years)') + ylab('Number of players') + ggtitle('Distribution of career length') + theme_calc()

data_explore %>% filter(Yrs<=5) %>% nrow()
data_explore %>% filter(Yrs>5) %>% nrow()

# Players drafted per year
# Population is represented more by players drafted before mid 90s.
ggplot(data_explore, aes(x=Year_drafted)) + geom_bar(fill = '#4BAEFA') + xlab('Year') +
  ylab('Number of players') + ggtitle('Distribution of drafting years') + theme_calc()



# Correlation matrix
data_explore %>% dplyr::select(-Name) %>%
  ggcorr(color = "grey50", hjust = 1, label = TRUE,
         label_alpha = 0.5, label_size = 4, label_round = 2, layout.exp = 3)


# Distribution of each variable - violin plots
colNames <- colnames(data_explore)[2:(ncol(data_explore)-1)]

plots_list_1 <- list()
colNames_1 <- c('MIN','PTS','FG_made','FGA','FT_made','FTA','OREB','DREB','REB','AST','STL','TOV','TP_made','TPA','BLK')

k <- 1
for (i in colNames_1) {
  i_mean <- mean(as_vector(data_explore[i]))
  i_median <- median(as_vector(data_explore[i]))
  plots_list_1[[k]] <- ggplot(data=data_explore, aes_string(y=i, x=1)) + geom_violin(fill = '#4BAEFA') + #fill = '#4BAEFA', color = 'black') +
    geom_hline(yintercept = i_mean, color = '#FF9A3D', size = 1.5) + # mean - orange line
    geom_hline(yintercept = i_median, color = '#FFE017', size = 1.5) + # median - yellow line
    xlab('') + theme_calc()+ theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
  k <- k + 1
}

do.call("grid.arrange", c(plots_list_1, ncol=5, nrow=3))

plots_list_2 <- list()
colNames_2 <- c('GP', 'FG_percent', 'TP_percent', 'FT_percent')

k <- 1
for (i in colNames_2) {
  i_mean <- mean(as_vector(data_explore[i]))
  i_median <- median(as_vector(data_explore[i]))
  plots_list_2[[k]] <- ggplot(data=data_explore, aes_string(y=i, x=1)) + geom_violin(fill = '#4BAEFA') + #fill = '#4BAEFA', color = 'black') +
    geom_hline(yintercept = i_mean, color = '#FF9A3D', size = 1.5) + # mean - orange line
    geom_hline(yintercept = i_median, color = '#FFE017', size = 1.5) + # median - yellow line
    xlab('') + theme_calc()+ theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
  k <- k + 1
}

do.call("grid.arrange", c(plots_list_2, ncol=2, nrow=2))


# BUILDING THE MODELS - LASSO REGRESSION-------------------------------
# Use glmnet for all models - preparation of training and test predictors
# and response variables.
x <- model.matrix(Target~.-1,data=data_train)
y <- data_train$Target
x.test <- model.matrix(Target~.-1,data=data_test)

# Fitting lasso regression with multiple lambda values
set.seed(321)
fit.lasso <- glmnet(x, y, alpha=1, family = 'binomial')
plot(fit.lasso,xvar="lambda",label=TRUE)

# Cross-validation of lasso regression to get the optimal labda value
set.seed(321)
cv.lasso <- cv.glmnet(x, y, alpha = 1, family = 'binomial')
plot(cv.lasso)

log(cv.lasso$lambda.min) # log-lambda with minimal binomial deviance -3.487281
log(cv.lasso$lambda.1se) # log-lambda within 1 standard error from minimal binomial deviance -2.649978

# fitting lasso regression with optimal lambda value (1 SE from minimal binomial deviance)
set.seed(321)
fit.lasso.min.lambda <- glmnet(x, y, alpha = 1, lambda = cv.lasso$lambda.1se, family = 'binomial')
coef(fit.lasso.min.lambda)

# computing predicted values using test dataset (as classess and probabilities of falling into classes)
predictions.class.lasso <- predict(fit.lasso.min.lambda, newx = x.test, type = "class") %>% as.factor()
predictions.probs.lasso <- predict(fit.lasso.min.lambda, newx = x.test, type = "response") %>% as.vector()

# Plotting ROC curve - package plotROC was preferred to the function in glment package as it
# provides prettier plots and easier way of AUC calculation.
rocpredict.lasso <- data.frame(D=as.numeric(data_test$Target)-1, M=predictions.probs.lasso)
rocplot.lasso <- ggplot(data=rocpredict.lasso, aes(m = M, d = D)) + geom_roc(n.cuts = 0)
rocplot.lasso + style_roc() + annotate('text', x = 0.75, y = 0.25,
                                   label = paste("AREA UNDER CURVE = ", round(calc_auc(rocplot.lasso)$AUC, 3)))
# Check ROC curve plot suing functions from glmnet package - same as the above
plot(roc.glmnet(fit.lasso.min.lambda, newx = x.test, newy=data_test$Target))

# Confusion matrix from caret package in addition to accuracy provides more metrics
# such as sensitivity and specificity, so it was preferred to confusion.glmnet functions from glmnet package
performance.lasso <- confusionMatrix(data=predictions.class.lasso, reference = data_test$Target, positive = 'S')
performance.lasso

# Check accuracy rate calculated by glmnet package functions - same as the above
confusion.glmnet(fit.lasso.min.lambda, newx = x.test, newy=data_test$Target)


# BUILDING THE MODELS - RIDGE REGRESSION---------------------------
# Fitting ridge regression with multiple lambda values
set.seed(321)
fit.ridge <- glmnet(x, y, alpha=0, family = 'binomial')
plot(fit.ridge,xvar="lambda",label=TRUE)

# Cross-validation of ridge regression to get the optimal labda value
set.seed(321)
cv.ridge <- cv.glmnet(x, y, alpha = 0, family = 'binomial')
plot(cv.ridge)

log(cv.ridge$lambda.min) # log-lambda with minimal binomial deviance -0.9521119
log(cv.ridge$lambda.1se) # log-lambda within 1 standard error from minimal binomial deviance 0.7224954

# fitting ridge regression with optimal lambda value (1 SE from minimal binomial deviance)
set.seed(321)
fit.ridge.min.lambda <- glmnet(x, y, alpha = 0, lambda = cv.ridge$lambda.1se, family = 'binomial')
coef(fit.ridge.min.lambda)

# computing predicted values using test dataset (as classess and probabilities of falling into classes)
predictions.class.ridge <- predict(fit.ridge.min.lambda, newx = x.test, type = "class") %>% as.factor()
predictions.probs.ridge <- predict(fit.ridge.min.lambda, newx = x.test, type = "response") %>% as.vector()

# Plotting ROC curve
rocpredict.ridge <- data.frame(D=as.numeric(data_test$Target)-1, M=predictions.probs.ridge)
rocplot.ridge <- ggplot(data=rocpredict.ridge, aes(m = M, d = D)) + geom_roc(n.cuts = 0)
rocplot.ridge + style_roc() + annotate('text', x = 0.75, y = 0.25,
                                   label = paste("AREA UNDER CURVE = ", round(calc_auc(rocplot.ridge)$AUC, 3)))

# Confusion matrix
performance.ridge <- confusionMatrix(data=predictions.class.ridge, reference = data_test$Target, positive = 'S')
performance.ridge

# BUILDING THE MODELS - ELASTIC NETS -------------------------------
# using caret package to get optimal alpha and lambda
predictors <- colnames(data_train)[1:20]
set.seed(321)
model.net <- train(
  Target ~.-1, data = data_train, method = 'glmnet', family = 'binomial',
  trControl = trainControl('cv', number=10, classProbs = TRUE, summaryFunction = twoClassSummary), #, summaryFunction = twoClassSummary
  tuneLength = 10,
  metric = "ROC"
)

# checking optimal tuning parameters and model coefficients with these parameters
model.net$bestTune #  alpha = 0.4 lambda = 0.06713159
coef(model.net$finalModel, model.net$bestTune$lambda)

# fitting ridge regression with optimal lambda and alpha values and
# computing predicted values using test dataset (as classess and probabilities of falling into classes)
set.seed(321)
modelbest.net <- glmnet(x, y, alpha = model.net$bestTune['alpha'], lambda = model.net$bestTune['lambda'], family = 'binomial')
predictions.class.net <- predict(modelbest.net, newx = x.test, type = "class") %>% as.factor()
predictions.probs.net <- predict(modelbest.net, newx = x.test, type = "response") %>% as.vector()

# Plotting ROC curve
rocpredict.net <- data.frame(D=as.numeric(data_test$Target)-1, M=predictions.probs.net)
rocplot.net <- ggplot(data=rocpredict.net, aes(m = M, d = D)) + geom_roc(n.cuts = 0)
rocplot.net + style_roc() + annotate('text', x = 0.75, y = 0.25,
                                   label = paste("AREA UNDER CURVE = ", round(calc_auc(rocplot.net)$AUC, 3)))

# Confusion matrix
performance.net <- confusionMatrix(data=predictions.class.net, reference = data_test$Target, positive = 'S')



# COMPARISON OF PERFORMANCE -------------------------------
# Combining ROC plots for the report
ROC.df <- tibble(Observed = as.numeric(data_test$Target)-1,
                 Lasso = predictions.probs.lasso,
                 Ridge = predictions.probs.ridge,
                 Net = predictions.probs.net)

ROC.colors <- c("Lasso regression" = "#0eba2d",
                "Ridge regression" = "#b8160e",
                "Elastic nets" = "#004EF5")

ROC.plot <- ggplot(ROC.df, aes(d = Observed)) +
  geom_roc(aes(m = Lasso, color = 'Lasso regression'), labels = FALSE, size = 2) +
  annotate('text', x = 0.60, y = 0.50, color = "#0eba2d",
           label = paste("Lasso regression AUC = ", round(calc_auc(rocplot.lasso)$AUC, 3))) +
  geom_roc(aes(m = Ridge, color = 'Ridge regression'), labels = FALSE, size = 2) +
  annotate('text', x = 0.60, y = 0.40, color = "#b8160e",
           label = paste("Ridge regression AUC = ", round(calc_auc(rocplot.ridge)$AUC, 3))) +
  geom_roc(aes(m = Net, color = 'Elastic nets'), labels = FALSE, size = 2, alpha = 0.3) +
  annotate('text', x = 0.60, y = 0.30, color = "#004EF5",
           label = paste("Elastic nets AUC = ", round(calc_auc(rocplot.net)$AUC, 3))) +
  style_roc() +
  labs(color = "Legend") +
  scale_color_manual(values = ROC.colors) +
  theme(legend.position="bottom")

ROC.plot


# Table with comparison of accuracy, specificity and sensitivity
perf.measurements <- rbind(Area_under_ROC_curve = c(round(calc_auc(rocplot.lasso)$AUC,3),
                                                    round(calc_auc(rocplot.ridge)$AUC,3),
                                                    round(calc_auc(rocplot.net)$AUC,3)),
                           Accuracy = c(round(performance.lasso$overall['Accuracy'],3),
                                        round(performance.ridge$overall['Accuracy'],3),
                                        round(performance.net$overall['Accuracy'],3)),
                           Sensitivity = c(round(performance.lasso$byClass['Sensitivity'],3),
                                           round(performance.ridge$byClass['Sensitivity'],3),
                                           round(performance.net$byClass['Sensitivity'],3)),
                           Specificity = c(round(performance.lasso$byClass['Specificity'],3),
                                           round(performance.ridge$byClass['Specificity'],3),
                                           round(performance.net$byClass['Specificity'],3)))
colnames(perf.measurements) <- c("Lasso regression", "Ridge regression", "Elastic nets")

perf.measurements

# accuracy, sensitivity and specificity of lasso model on training data to check signs of overfit
predictions.class.lasso.train <- predict(fit.lasso.min.lambda, newx = x, type = "class") %>% as.factor()
predictions.probs.lasso.train <- predict(fit.lasso.min.lambda, newx = x, type = "response") %>% as.vector()

# Calculation AUC.
rocpredict.lasso.train <- data.frame(D=as.numeric(data_train$Target)-1, M=predictions.probs.lasso.train)
rocplot.lasso.train <- ggplot(data=rocpredict.lasso.train, aes(m = M, d = D)) + geom_roc(n.cuts = 0)
round(calc_auc(rocplot.lasso.train)$AUC, 3)

# Calculating accuracy, sensitivity and specificity
performance.lasso.train <- confusionMatrix(data=predictions.class.lasso.train, reference = data_train$Target, positive = 'S')
performance.lasso.train
