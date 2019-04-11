require(xgboost)

# load in the agaricus dataset

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)


watchlist <- list(eval = dtest, train = dtrain)
num_round <- 20


logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  grad   <- ifelse(preds > labels,0.5*(preds-labels),-2*(labels-preds))
  hess   <- ifelse(preds > labels,0.5,2)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err    <- max(preds - labels,0.5*(labels - preds))^2
  return(list(metric = "error", value = err))
}

param <- list(max_depth=2, eta=1, nthread = 2, verbosity=0, 
              objective=logregobj, eval_metric=evalerror)
print ('start training with user customized objective')

bst <- xgb.train(param, dtrain, num_round, watchlist)
