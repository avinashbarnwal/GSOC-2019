library(xgboost)
setwd('')

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')



dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

