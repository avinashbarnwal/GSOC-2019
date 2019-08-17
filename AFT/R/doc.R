library(xgboost)
require(stringr)
setwd('/Users/avinashbarnwal/Desktop/Personal/GSOC-2019/AFT/')

inputs    = read.table('test/data/neuroblastoma-data-master/data/ATAC_JV_adipose/inputs.csv',sep=",",header=T,stringsAsFactors = FALSE)
naColumns = colnames(inputs)[colSums(is.na(inputs))>0]
inputs    = inputs[ , !(names(inputs) %in% naColumns)]
#inputs['sequenceID'] = str_trim(inputs['sequenceID'])

labels     = read.table('test/data/neuroblastoma-data-master/data/ATAC_JV_adipose/outputs.csv',sep=",",header=T,stringsAsFactors = FALSE)
folds      = read.table('test/data/neuroblastoma-data-master/data/ATAC_JV_adipose/cv/equal_labels/folds.csv',sep=",",header=T,stringsAsFactors = FALSE) 
folds_iter = unique(folds$fold)

for(i in c(1)){
  test_fold    = i
  test_id      = folds[folds$fold==i,'sequenceID']
  train_id     = folds[folds$fold!=i,'sequenceID']
  X            = inputs[inputs$sequenceID %in% train_id,]
  X            = X[,-which(names(X) %in% c("sequenceID"))]
  X            = as.matrix(X)
  X_val        = inputs[inputs$sequenceID %in% test_id,]
  X_val        = X_val[,-which(names(X_val) %in% c("sequenceID"))]
  X_val        = as.matrix(X_val)
  labels$min.log.lambda = lapply(labels$min.log.lambda,exp)
  labels$max.log.lambda = lapply(labels$max.log.lambda,exp)
  
  y_label       = labels[labels$sequenceID %in% train_id,]
  y_label_test  = labels[labels$sequenceID %in% test_id,]
  y_lower       = as.matrix(y_label$min.log.lambda)
  y_higher      = as.matrix(y_label$max.log.lambda)
  y_lower_val   = as.matrix(y_label_test$min.log.lambda)
  y_higher_val  = as.matrix(y_label_test$max.log.lambda)
  
  dtrain = xgb.DMatrix(X)
  setinfo(dtrain,'label_lower_bound', y_lower)
  setinfo(dtrain,'label_higer_bound', y_higher)
  
  dtest = xgb.DMatrix(X_val)
  setinfo(dtest,'label_lower_bound', y_lower_val)
  setinfo(dtest,'label_higer_bound', y_higher_val)
  
  watchlist <- list(eval = dtest, train = dtrain)
  num_round <- 20
  param     <- list(learning_rate=0.1, aft_noise_distribution='normal', 
                    nthread = 2, verbosity=0, aft_sigma= 1.0,
                    eval_metric  = 'aft-nloglik@normal,1.0',
                    objective  = "aft:survival")
  bst       <- xgb.train(param, dtrain, num_round, watchlist)
  
} 



