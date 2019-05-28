library(tidyverse)
library(survival)
library(ggplot2)
library(latex2exp)
set.seed(2)
# Get 1,000 observation
b0 <- 17
b1 <- 0.5
id <- 1:1000
x1 <- runif(1000, min = -100, 100)
sigma <- 4
eps <- rnorm(1000, mean = 0, sigma)
df  <- data.frame(cbind(id,x1,eps))

# Set up the data
df['y'] = b0 + b1*df['x1'] + eps
# Convert all negative to zero
df[df['y']<=0,'y'] = 0
# Convert first 500 obs > 50 to 50
df['y_cen'] = df['y']
df[df['y']>50 & df['id'] < 500,'y_cen'] = 50
# Convert last 500 obs > 40 to 40
df[df['y']>40 & df['id'] > 500,'y_cen'] = 40

# Define left and rigth variables
df['left']  = df['y']
df['right'] = df['y']

df[df['y']<=0,'left'] = -Inf
df[(df['y']>50 & df['id'] < 500) | (df['y']>40 & df['id'] > 500),'right'] = Inf
n = nrow(df)

for (i in 1:n){
  if(df[i,'right']>=10 && df[i,'right']<=20){
    df[i,'right'] = df[i,'right'] +3
  }
}



for (i in 1:n){
  if(df[i,'right']==0){
    df[i,'right'] = df[i,'right'] + 20 
  }
}



res_gaussian <- survreg(Surv(left, right, type = "interval2") ~ x1,
                       data = df, dist = "gaussian")

res_logistic <- survreg(Surv(left, right, type = "interval2") ~ x1,
                        data = df, dist = "logistic")

summary(res)

#left #right #interval #point
#1    #2     #3        #8

#Loss Formula for Log Normal
#left censored  - 1/2(1+erf(log(t/t^)/sigma\sqrt2))
#right censored - 1/2+erf(log(t/t^)/sigma\sqrt2)
#interval - 1/2(erf(log(t_higher/t^)/sigma\sqrt2) - erf(log(t_lower/t^)/sigma\sqrt2))
#https://www.mathworks.com/matlabcentral/answers/428624-cdf-for-loglogistic-distribution
#https://en.wikipedia.org/wiki/Log-normal_distribution

t      = 1
y_pred = 0.5
sigma  = 1

loss_lognormal <- function(type="left",t_event=NULL,t_lower=NULL,t_higher=NULL,y_pred=1){
  if(type=="uncensored"){
    log_normal_uncensored = -log(1/(y_pred*sigma*sqrt(2*3.14))*exp((log(t_event/y_pred))**2/(-2*sigma*sigma)))
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

loss_loglogistic <- function(type="left",t_event = NULL,t_lower=NULL,t_higher=NULL,y_pred=1){
  
  if(type=="uncensored"){
    log_logistic_uncensored = -log((1/(sigma*y_pred))*exp(log(t_event/y_pred)/sigma)/(1+exp(log(t_event/y_pred)/sigma))**2)
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


generateLognormal<-function(type="left",t_event = NULL,t_lower=NULL,t_higher=NULL){
  y    = seq(1,10)
  loss = vector()
  for( i in y){
    loss[i] = loss_lognormal(type=type,t_event=t_event,t_lower=t_lower,t_higher=t_higher,y_pred=exp(i))
  }
  return(loss)
}


generateLoglogistic<-function(type="left",t_event=NULL,t_lower=NULL,t_higher=NULL){
  y    = seq(1,10)
  loss = vector()
  for( i in y){
    loss[i] = loss_loglogistic(type=type,t_event=t_event,t_lower=t_lower,t_higher=t_higher,y_pred=exp(i))
  }
  return(loss)
}

sim_loss_lognormal_left = generateLognormal(type="left",t_higher=20)
sim_loss_loglistic_left = generateLoglogistic(type="left",t_higher=20)

ggplot()+geom_line(aes(x=y,y=sim_loss_lognormal_left),color='red')+geom_line(aes(x=y,y=sim_loss_loglistic_left),color='green')+
labs(x = "y predicted in log scale",y = "loss function",title='Left Censored')+geom_vline(xintercept=log(20))

sim_loss_lognormal_right = generateLognormal(type="right",t_lower=60)
sim_loss_loglistic_right = generateLoglogistic(type="right",t_lower=60)

ggplot()+geom_line(aes(x=y,y=sim_loss_lognormal_right),color='red')+geom_line(aes(x=y,y=sim_loss_loglistic_right),color='green')+
  labs(x = "y predicted in log scale",y = "loss function",title='Right Censored')+geom_vline(xintercept=log(60))

sim_loss_lognormal_interval = generateLognormal(type="interval",t_lower=16,t_higher=20)
sim_loss_loglistic_interval = generateLoglogistic(type="interval",t_lower=16,t_higher=20)

ggplot()+geom_line(aes(x=y,y=sim_loss_lognormal_interval),color='red')+geom_line(aes(x=y,y=sim_loss_loglistic_interval),color='green')+
  labs(x = "y predicted in log scale",y = "loss function",title='Interval Censored')+geom_vline(xintercept=log(16))+geom_vline(xintercept=log(20))

sim_loss_lognormal_uncensored = generateLognormal(type="uncensored",t_event=40)
sim_loss_loglistic_uncensored = generateLoglogistic(type="uncensored",t_event=40)

ggplot()+geom_line(aes(x=y,y=sim_loss_lognormal_uncensored),color='red')+geom_line(aes(x=y,y=sim_loss_loglistic_uncensored),color='green')+
  labs(x = "y predicted in log scale",y = "loss function",title='UnCensored')+geom_vline(xintercept=log(40))

