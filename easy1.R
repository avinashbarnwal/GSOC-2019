# Check package availability
is.installed <- function(mypkg){
  is.element(mypkg, installed.packages()[,1])
} 


if (!is.installed("xgboost")){
  install.packages("xgboost")
}

if (!is.installed("ggplot2")){
  install.packages("ggplot2")
}

if (!is.installed("MLmetrics")){
  install.packages("MLmetrics")
}

if (!is.installed("kableExtra")){
  install.packages("kableExtra")
}


library(xgboost)
library(ggplot2)
library(MLmetrics)
library(knitr)
library(kableExtra)

set.seed(123)

#Reference for Count regression
#https://newonlinecourses.science.psu.edu/stat504/node/169/

data_link = "https://newonlinecourses.science.psu.edu/stat504/sites/onlinecourses.science.psu.edu.stat504/files/lesson07/crab/index.txt"
crab      = read.table(data_link,stringsAsFactors = FALSE,header=FALSE)
colnames(crab)=c("Obs","C","S","W","Wt","Sa")
#### to remove the column labeled "Obs"
crab      = crab[,-1]
str(crab)


ggplot(data = crab) +
  geom_histogram(mapping = aes(x = Sa), binwidth = 0.5)+labs(title="Frequency of Sa Variable")

train_test_split <- function(data,frac=0.8){
  
  ## 75% of the sample size
  smp_size  <- floor(frac * nrow(data))
  ## set the seed to make your partition reproducible
  train_ind <- sample(seq_len(nrow(data)), size = smp_size)
  train     <- data[train_ind, ]
  test      <- data[-train_ind, ]
  
  result       <- list()
  result$train <- train
  result$test  <- test
  
  return(result)
  
}

result   <-  train_test_split(crab,frac = 0.8)
train    <-  as.matrix(result$train)
test     <-  as.matrix(result$test)


#Testing with eta = 0.02, max_depth = 2,min_child_weight = 5, subsample = 0.5

df_train  <- xgb.DMatrix(train[,c(1,2,3,4)], label = train[,5])
df_test   <- xgb.DMatrix(test[,c(1,2,3,4)],  label = test[,5])

paramGrid <- expand.grid(subsample <- c(0.5, 0.75, 1), 
                         eta       <- c(0.01,0.05,0.01),
                         max.depth <- c(1,3,1))

colnames(paramGrid) <- c("subsample","eta","max.depth")

GridSearchCV        <- apply(paramGrid, 1, function(parameterList){
  
  ##Extract Parameters to test#
  param      <- list(objective        = "count:poisson",                 # Count Regression
                     eta              = parameterList[["eta"]],          # No overfitting
                     max.depth        = parameterList[["max.depth"]],    # No overfitting
                     subsample        = parameterList[["subsample"]],    # No overfitting
                     nthread          = 4)
  
  
  n_fold = 5
  nTrees = 100
  fit               <- xgb.cv(data = df_train, params=param, nfold = n_fold, nrounds = nTrees,    showsd = F,verbose = F)
  validError        <- fit$evaluation_log$test_poisson_nloglik_mean
  #Save rmse of the last iteration
  #return(c(validError, subsample, eta, max.depth))
  return(validError)
})

min_index       = which.min(apply(GridSearchCV,2,min))
best_param_grid = paramGrid[min_index,]

best_param          <- list(objective        = "count:poisson",                 
                            eta              = as.numeric(best_param_grid['eta']),
                            max.depth        = as.numeric(best_param_grid['max.depth']),
                            subsample        = as.numeric(best_param_grid['subsample']),
                            nthread          = 4)


xgb_model          <-  xgb.train(best_param,df_train,nrounds=100)
model_y_pred       <-  predict(xgb_model,df_test)

test_error         <- Poisson_LogLoss(y_pred = model_y_pred, test[,5])
base_y_pred        <- rep(mean(train[,5]),nrow(test))
base_test_error    <- Poisson_LogLoss(y_pred = base_y_pred, test[,5])

n_test                 <- length(test[,5])
df_result              <- data.frame(seq(1,n_test),test[,5], model_y_pred, base_y_pred)
colnames(df_result)    <- c("Obs","actual_y","model_y","base_y")
result_error           <- data.frame(Error=c(test_error,base_test_error))
rownames(result_error) <- c("Model Error","Base Error") 

#Plotting
ggplot(df_result, aes(x = Obs, y = actual_y))                + 
geom_line(aes(y = actual_y,     colour = "Actual Y"))        +  
geom_line(aes(y = model_y,      colour = "Predicted Y"))     +
geom_line(aes(y = base_y_pred,  colour = "Base Y"))          + 
xlab('Observation Number') + ylab('Y') +
ggtitle('Model Error')  



