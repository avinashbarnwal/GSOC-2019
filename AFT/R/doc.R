require(xgboost)
setwd('/Users/avinashbarnwal/Desktop/Personal/GSOC-2019/AFT/')

data_import <-function(){
  
  inputs = read.table('https://raw.githubusercontent.com/avinashbarnwal/GSOC-2019/master/AFT/test/data/neuroblastoma-data-master/data/ATAC_JV_adipose/inputs.csv',sep=",",header=T,stringsAsFactors = F)
  labels = read.table('https://raw.githubusercontent.com/avinashbarnwal/GSOC-2019/master/AFT/test/data/neuroblastoma-data-master/data/ATAC_JV_adipose/outputs.csv',sep=",",header=T,stringsAsFactors = F)
  folds  = read.table('https://raw.githubusercontent.com/avinashbarnwal/GSOC-2019/master/AFT/test/data/neuroblastoma-data-master/data/ATAC_JV_adipose/cv/equal_labels/folds.csv',sep=",",header=T,stringsAsFactors = F) 

  res        = list()
  res$inputs = inputs
  res$labels = labels
  res$folds  = folds
  
  return(res)
}

data_massage <- function(inputs,labels){
  
  naColumns = colnames(inputs)[colSums(is.na(inputs))>0]
  inputs    = inputs[ , !(names(inputs) %in% naColumns)]
  labels$min.log.lambda = lapply(labels$min.log.lambda,exp)
  labels$max.log.lambda = lapply(labels$max.log.lambda,exp)
  res = list()
  res$inputs = inputs
  res$labels = labels
  return(res)
}

getXY<-function(foldNo,folds,inputs,labels){
  
  test_id      = folds[folds$fold==foldNo,'sequenceID']
  train_id     = folds[folds$fold!=foldNo,'sequenceID']
  X            = inputs[inputs$sequenceID %in% train_id,]
  X            = X[,-which(names(X) %in% c("sequenceID"))]
  X            = as.matrix(X)
  X_val        = inputs[inputs$sequenceID %in% test_id,]
  X_val        = X_val[,-which(names(X_val) %in% c("sequenceID"))]
  X_val        = as.matrix(X_val)
  
  y_label       = labels[labels$sequenceID %in% train_id,]
  y_label_test  = labels[labels$sequenceID %in% test_id,]
  y_lower       = as.matrix(y_label$min.log.lambda)
  y_upper       = as.matrix(y_label$max.log.lambda)
  y_lower_val   = as.matrix(y_label_test$min.log.lambda)
  y_upper_val   = as.matrix(y_label_test$max.log.lambda)
  
  res   = list()
  res$X = X
  res$X_val    = X_val
  
  res$y_lower      = y_lower
  res$y_lower_val  = y_lower_val
  
  res$y_upper     = y_upper
  res$y_upper_val = y_upper_val
  
  return(res)
}

getParam <- function(sigma,distribution,learning_rate){
  eval_metric = paste("aft-nloglik@",distribution,",",sigma,sep="") 
  param       = list(learning_rate=learning_rate, aft_noise_distribution=distribution, 
                    nthread = 4, verbosity=0, aft_sigma= sigma,
                    eval_metric  = eval_metric,
                    objective  = "aft:survival")
  
  return(param)
}

trainModel <- function(foldNo,X,X_val,y_lower,y_lower_val,y_upper,y_upper_val,param,num_round){
  
  dtrain = xgb.DMatrix(X)
  setinfo(dtrain,'label_lower_bound', y_lower)
  setinfo(dtrain,'label_upper_bound', y_upper)
  
  dtest = xgb.DMatrix(X_val)
  setinfo(dtest,'label_lower_bound', y_lower_val)
  setinfo(dtest,'label_upper_bound', y_upper_val)
  
  watchlist <- list(eval = dtest, train = dtrain)
  param     <- getParam(sigma,distribution,learning_rate)
  bst       <- xgb.train(param, dtrain, num_round, watchlist)
  
  return(bst)
}


res    = data_import()
inputs = res$inputs
labels = res$labels
folds  = res$folds

res    = data_massage(inputs,labels)
inputs = res$inputs
labels = res$labels
#Set Parameters
sigma         = 100.0
distribution  = 'normal'
learning_rate = 0.1
num_round     = 50
folds_iter    = unique(folds$fold)

  
for(i in folds_iter){
  
  res           = getXY(i,folds,inputs,labels)
  X             = res$X
  X_val         = res$X_val
  y_lower       = res$y_lower
  y_lower_val   = res$y_lower_val
  y_upper      = res$y_upper
  y_upper_val  = res$y_upper_val

  param = getParam(sigma,distribution,learning_rate)
  bst   = trainModel(i,X,X_val,y_lower,y_lower_val,y_upper,y_upper_val,param,num_round)
  
} 



