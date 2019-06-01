library(tidyverse)
library(survival)
library(ggplot2)
library(latex2exp)
library(tikzDevice)

#http://home.iitk.ac.in/~kundu/paper146.pdf

set.seed(2)
sigma    = 1
n.points = 15

loss_lognormal <- function(type="left",t.lower=NULL,t.higher=NULL,y.hat=1){
  
  n.points      = length(y.hat)
  t.lower.col   = rep(t.lower,n.points)
  t.higher.col  = rep(t.higher,n.points)
  dist_type     = rep("Normal",n.points)
  
  if(type=="uncensored"){
    
    cost  = -log(1/(y.hat*sigma*sqrt(2*pi))*exp((log(t.lower/y.hat))**2/(-2*sigma*sigma)))
    data_type     = rep("Uncensored",n.points)
    data = data.frame(y.hat = y.hat,cost=cost,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  } 
  else if(type=="left"){
    cost       = -log(pnorm(log(t.higher/y.hat)/sigma))
    data_type     = rep("Left",n.points)
    data = data.frame(y.hat = y.hat,cost=cost,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  else if(type=="right"){
    cost      = -log(1-pnorm(log(t.lower/y.hat)/sigma))
    data_type     = rep("Right",n.points)
    data = data.frame(y.hat = y.hat,cost=cost,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  else{
    cost   = -log(pnorm(log(t.higher/y.hat)/(sigma)) - pnorm(log(t.lower/y.hat)/(sigma)))
    data_type     = rep("Interval",n.points)
    data = data.frame(y.hat = y.hat,cost=cost,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
}


loss_loglogistic <- function(type="left",t.lower=NULL,t.higher=NULL,y.hat=1){
  
  n.points      = length(y.hat)
  t.lower.col   = rep(t.lower,n.points)
  t.higher.col  = rep(t.higher,n.points)
  dist_type     = rep("Logistic",n.points)
  
  if(type=="uncensored"){
    cost = -log((1/(sigma*t.lower))*exp(log(t.lower/y.hat)/sigma)/(1+exp(log(t.lower/y.hat)/sigma))**2)
    data_type     = rep("Uncensored",n.points)
    data = data.frame(y.hat = y.hat,cost=cost,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  else if(type=="left"){
    cost       = -log(exp(log(t.higher/y.hat)/sigma)/(1+exp(log(t.higher/y.hat)/sigma)))
    data_type     = rep("Left",n.points)
    data = data.frame(y.hat = y.hat,cost=cost,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  else if(type=="right"){
    cost      = -log(1- exp(log(t.lower/y.hat)/sigma)/(1+exp(log(t.lower/y.hat)/sigma)))
    data_type     = rep("Right",n.points)
    data = data.frame(y.hat = y.hat,cost=cost,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  else{
    cost   = -log(exp(log(t.higher/y.hat)/sigma)/(1+exp(log(t.higher/y.hat)/sigma)) - exp(log(t.lower/y.hat)/sigma)/(1+exp(log(t.lower/y.hat)/sigma)))
    data_type     = rep("Interval",n.points)
    data = data.frame(y.hat = y.hat,cost=cost,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
}

distribution.list <- list(gaussian=list(uncensored=loss_lognormal(type="uncensored",t.lower=100,t.higher=100,y.hat=2**(seq(1,10,length=15))),
                                        left=loss_lognormal(type="left",t.lower=-Inf,t.higher=20,y.hat=2**(seq(1,10,length=15))),
                                        right=loss_lognormal(type="right",t.lower=60,t.higher=Inf,y.hat=2**(seq(1,10,length=15))),
                                        interval=loss_lognormal(type="interval",t.lower=16,t.higher=200,y.hat=2**(seq(1,10,length=15)))
                                        ))

for(distribution in names(distribution.list)){
  loss.fun.list <- distribution.list[[distribution]]
  for(type in names(loss.fun.list)){
    data_complete_list[[paste(distribution, type)]] <- loss.fun.list[[type]]
  }
}

data_complete <- do.call(rbind, data_complete_list)


n.points       = 15
y              = 2**(seq(1,10,length=n.points))
t_lower_left   = rep(-Inf,n.points)
t_higher_left  = rep(20,n.points)
type_left      = rep("Left",n.points)

sim_loss_lognormal_left = generateLognormal(type="left",t.lower = -Inf,t.higher=20)
sim_loss_loglistic_left = generateLoglogistic(type="left",t.lower = -Inf,t.higher=20)

data_left = data.frame(y = y,LogNormal = sim_loss_lognormal_left,LogLogistic = sim_loss_loglistic_left,t_lower=t_lower_left,t_higher=t_higher_left,type=type_left)

sim_loss_lognormal_right = generateLognormal(type="right",t.lower=60,t.higher=Inf)
sim_loss_loglistic_right = generateLoglogistic(type="right",t.lower=60,t.higher=Inf)


t.lower.right  = rep(60,n.points)
t.higher.right = rep(Inf,n.points)
type.right     = rep("Right",n.points)

data_right = data.frame(y = y,LogNormal = sim_loss_lognormal_right,LogLogistic = sim_loss_loglistic_right,t.lower=t.lower.right,t_higher=t_higher_right,type=type_right)

sim_loss_lognormal_uncensored = generateLognormal(type="uncensored",t_lower=100,t_higher=100)
sim_loss_loglistic_uncensored = generateLoglogistic(type="uncensored",t_lower=100,t_higher=100)

t_lower_uncensored   = rep(100,n.points)
t_higher_uncensored  = rep(100,n.points)
type_uncensored      = rep("Uncensored",n.points)

data_uncensored = data.frame(y = y,LogNormal = sim_loss_lognormal_uncensored,LogLogistic = sim_loss_loglistic_uncensored,t_lower=t_lower_uncensored,t_higher=t_higher_uncensored,type=type_uncensored)

sim_loss_lognormal_interval = generateLognormal(type="interval",t_lower=16,t_higher=200)
sim_loss_loglistic_interval = generateLoglogistic(type="interval",t_lower=16,t_higher=200)
t_lower_interval   = rep(16,n.points)
t_higher_interval  = rep(200,n.points)
type_interval      = rep("Interval",n.points)

data_interval = data.frame(y = y,LogNormal = sim_loss_lognormal_interval,LogLogistic = sim_loss_loglistic_interval,t_lower=t_lower_interval,t_higher=t_higher_interval,type=type_interval)
data_complete = rbind(data_left,data_right,data_uncensored,data_interval)

p <- ggplot(data=data_complete) +
     geom_line(aes(x=y,y=LogNormal,colour="green"),
              data=data_complete,size=1) + scale_x_continuous(trans='log2') +
     geom_line(aes(y,LogLogistic,colour="red"),
              data=data_complete,size=1) + 
     geom_point(aes(t_lower,y=0),data=data_complete) +
     geom_point(aes(t_higher,y=0),data=data_complete)+ theme(legend.position=c(0.1,0.8))+
     ylab("loss function L_i(y_pred)")+
     xlab("predicted survival time y_pred in days (log_2 scale)") + facet_grid(. ~ type,scales="free") +
     scale_color_discrete(name = "Distribution", labels = c("Logistic", "Normal"))

p


