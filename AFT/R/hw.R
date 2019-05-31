library(tidyverse)
library(survival)
library(ggplot2)
library(latex2exp)
library(tikzDevice)

set.seed(2)
sigma  = 1

loss_lognormal <- function(type="left",t_lower=NULL,t_higher=NULL,y_pred=1){
  if(type=="uncensored"){
    log_normal_uncensored = -log(1/(y_pred*sigma*sqrt(2*3.14))*exp((log(t_lower/y_pred))**2/(-2*sigma*sigma)))
  } 
  else if(type=="left"){
    log_normal_left       = -log(pnorm(log(t_higher/y_pred)/sigma))
    return(log_normal_left)
  }
  else if(type=="right"){
    log_normal_right      = -log(1-pnorm(log(t_lower/y_pred)/sigma))
    return(log_normal_right)
  }
  else{
    log_normal_interval   = -log(pnorm(log(t_higher/y_pred)/(sigma)) - pnorm(log(t_lower/y_pred)/(sigma)))
    return(log_normal_interval)
  }
}

loss_loglogistic <- function(type="left",t_lower=NULL,t_higher=NULL,y_pred=1){
  
  if(type=="uncensored"){
    log_logistic_uncensored = -log((1/(sigma*t_lower))*exp(log(t_lower/y_pred)/sigma)/(1+exp(log(t_lower/y_pred)/sigma))**2)
    return(log_logistic_uncensored)
  }
  
  else if(type=="left"){
    log_logistic_left       = -log(exp(log(t_higher/y_pred)/sigma)/(1+exp(log(t_higher/y_pred)/sigma)))
    return(log_logistic_left)
  }
  
  else if(type=="right"){
    log_logistic_right      = -log(1- exp(log(t_lower/y_pred)/sigma)/(1+exp(log(t_lower/y_pred)/sigma)))
    return(log_logistic_right)
  }
  
  else{
    log_logistic_interval   = -log(exp(log(t_higher/y_pred)/sigma)/(1+exp(log(t_higher/y_pred)/sigma)) - exp(log(t_lower/y_pred)/sigma)/(1+exp(log(t_lower/y_pred)/sigma)))
    return(log_logistic_interval)
  }
}

generateLognormal<-function(type="left",t_lower=NULL,t_higher=NULL){
  y    = seq(1,10)
  loss = vector()
  for( i in y){
    loss[i] = loss_lognormal(type=type,t_lower=t_lower,t_higher=t_higher,y_pred=exp(i))
  }
  return(loss)
}


generateLoglogistic<-function(type="left",t_lower=NULL,t_higher=NULL){
  y    = seq(1,10)
  loss = vector()
  for( i in y){
    loss[i] = loss_loglogistic(type=type,t_lower=t_lower,t_higher=t_higher,y_pred=exp(i))
  }
  return(loss)
}

y              = seq(1,10)
t_lower_left   = rep(-Inf,10)
t_higher_left  = rep(log(20),10)
type_left      = rep("Left",10)

sim_loss_lognormal_left = generateLognormal(type="left",t_lower = -Inf,t_higher=20)
sim_loss_loglistic_left = generateLoglogistic(type="left",t_lower = -Inf,t_higher=20)

data_left = data.frame(y = y,LogNormal = sim_loss_lognormal_left,LogLogistic = sim_loss_loglistic_left,t_lower=t_lower_left,t_higher=t_higher_left,type=type_left)

sim_loss_lognormal_right = generateLognormal(type="right",t_lower=60,t_higher=Inf)
sim_loss_loglistic_right = generateLoglogistic(type="right",t_lower=60,t_higher=Inf)
t_lower_right   = rep(log(60),10)
t_higher_right  = rep(Inf,10)
type_right      = rep("Right",10)

data_right = data.frame(y = y,LogNormal = sim_loss_lognormal_right,LogLogistic = sim_loss_loglistic_right,t_lower=t_lower_right,t_higher=t_higher_right,type=type_right)

sim_loss_lognormal_uncensored = generateLognormal(type="uncensored",t_lower=100,t_higher=100)
sim_loss_loglistic_uncensored = generateLoglogistic(type="uncensored",t_lower=100,t_higher=100)
t_lower_uncensored   = rep(log(100),10)
t_higher_uncensored  = rep(log(100),10)
type_uncensored      = rep("Uncensored",10)

data_uncensored = data.frame(y = y,LogNormal = sim_loss_lognormal_uncensored,LogLogistic = sim_loss_loglistic_uncensored,t_lower=t_lower_uncensored,t_higher=t_higher_uncensored,type=type_uncensored)

sim_loss_lognormal_interval = generateLognormal(type="interval",t_lower=16,t_higher=200)
sim_loss_loglistic_interval = generateLoglogistic(type="interval",t_lower=16,t_higher=200)
t_lower_interval   = rep(log(16),10)
t_higher_interval  = rep(log(200),10)
type_interval      = rep("Interval",10)

data_interval = data.frame(y = y,LogNormal = sim_loss_lognormal_interval,LogLogistic = sim_loss_loglistic_interval,t_lower=t_lower_interval,t_higher=t_higher_interval,type=type_interval)
data_complete = rbind(data_left,data_right,data_uncensored,data_interval)

p <- ggplot(data=data_complete)+
  geom_line(aes(y,LogNormal,colour="green"),
            data=data_complete,size=1)+
  geom_line(aes(y,LogLogistic,colour="red"),
            data=data_complete,size=1)+
  geom_point(aes(t_lower,y=0),data=data_complete)+
  geom_point(aes(t_higher,y=0),data=data_complete)+xlim(0, 10)+ theme(legend.position=c(0.1,0.8))+
  ylab("loss function L_i(y_pred)")+
  xlab("predicted survival time y_pred in days (log_e scale)") + facet_grid(. ~ type,scales="free_y")+
  scale_color_discrete(name = "Distribution", labels = c("LogLogistic", "LogNormal"))

p


