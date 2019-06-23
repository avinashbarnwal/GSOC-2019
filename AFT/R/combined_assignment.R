set.seed(2)
library(ggplot2)
setwd('/Users/avinashbarnwal/Desktop/Personal/GSOC-2019/AFT/R')


z <- function(t.obs=t.obs,y.hat=y.hat,sigma=sigma){
  z = (log(t.obs)-log(y.hat))/sigma
  return(z)
}

f_z <- function(z=z,dist='logistic'){
  if(dist=='logistic'){
    f_z = exp(1)**z/(1+exp(1)**z)^2
  }
  if(dist=='normal'){
    f_z = dnorm(z)
  }
  return(f_z)
}

grad_f_z <- function(z=z,dist='logistic'){
  f_z      = f_z(z,dist)
  if(dist == 'logistic'){
    grad_f_z = f_z*(1-exp(1)**z)/(1+exp(1)**z)
  }
  if(dist == 'normal'){
    grad_f_z = -z*f_z
  }
  return(grad_f_z)
}

hes_f_z <- function(z=z,dist='logistic'){
  f_z = f_z(z,dist)
  if(dist=='normal'){
    hes_f_z = (z**2-1)*f_z    
  }
  if(dist=='logistic'){
    w       = exp(1)**z
    hes_f_z = f_z*(w**2-4*w+1)/(1+w)**2 
  }
  return(hes_f_z)
}

F_z <- function(z,dist='logistic'){
  if(dist=='logistic'){
    F_z = exp(1)**z/(1+exp(1)**z)
  }
  if(dist=='normal'){
    F_z = pnorm(z)
  }
  return(F_z)
}

grad_F_z <- function(z,dist){
  grad_F_z <- f_z(z,dist)
  return(grad_F_z)
}

f_y <- function(z=z,t.obs=t.obs,sigma=sigma,dist='logistic'){
  f_y = f_z(z,dist)/(t.obs*sigma)
  return(f_y)
}

grad_f_y <- function(z=z,t.obs=t.obs,sigma=sigma,dist='logistic'){
  grad_f_y = -grad_f_z(z,dist)/(sigma^2*t.obs)
  return(grad_f_y)
}  


loss <- function(type="left",t.lower=NULL,t.higher=NULL,sigma=1,y.hat=1,dist='normal'){
  
  n.points      = length(y.hat)
  t.lower.col   = rep(t.lower,n.points)
  t.higher.col  = rep(t.higher,n.points)
  dist_type     = rep(dist,n.points)
  
  if(type=="uncensored"){
    z     = (log(t.lower) - log(y.hat))/sigma  
    f_z   = f_z(z,dist)
    cost  = -log(f_z/(sigma*t.lower))
    data_type     = rep("Uncensored",n.points)
    parameter_type = rep('Loss',n.points)
    data          = data.frame(y.hat = y.hat,parameter=cost,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  } 
  else if(type=="left"){
    z             = (log(t.higher) - log(y.hat))/sigma
    F_z           = F_z(z,dist)
    cost          = -log(F_z)
    data_type     = rep("Left",n.points)
    parameter_type = rep('Loss',n.points)
    data          = data.frame(y.hat = y.hat,parameter=cost,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  else if(type=="right"){
    z             = (log(t.lower) - log(y.hat))/sigma
    F_z           = F_z(z,dist)
    cost          = -log(1-F_z)
    data_type     = rep("Right",n.points)
    parameter_type = rep('Loss',n.points)
    data          = data.frame(y.hat = y.hat,parameter=cost,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  else{
    z_u    = (log(t.higher) - log(y.hat))/sigma
    z_l    = (log(t.lower) - log(y.hat))/sigma
    F_z_u  = F_z(z_u,dist)
    F_z_l  = F_z(z_l,dist)
    cost   = -log(F_z_u - F_z_l)
    data_type     = rep("Interval",n.points)
    parameter_type = rep('Loss',n.points)
    data          = data.frame(y.hat = y.hat,parameter=cost,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
}

neg_grad <- function(type="left",t.lower=NULL,t.higher=NULL,sigma=1,y.hat=1,dist='normal'){
  
  n.points      = length(y.hat)
  t.lower.col   = rep(t.lower,n.points)
  t.higher.col  = rep(t.higher,n.points)
  dist_type     = rep(dist,n.points)
  if(type=="uncensored"){
    z             = (log(t.lower) - log(y.hat))/sigma  
    f_z           = f_z(z,dist)
    neg_grad      = -grad_f_z(z,dist)/(sigma*f_z)
    data_type     = rep("Uncensored",n.points)
    parameter_type = rep('Negative Gradient',n.points)
    data          = data.frame(y.hat = y.hat,parameter=neg_grad,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="left"){
    z              = (log(t.higher) - log(y.hat))/sigma
    f_z            = f_z(z,dist)
    neg_grad       = -f_z/(sigma*F_z(z,dist))
    data_type      = rep("Left",n.points)
    parameter_type = rep('Negative Gradient',n.points)
    data           = data.frame(y.hat = y.hat,parameter=neg_grad,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="right"){
    z              = (log(t.lower) - log(y.hat))/sigma
    f_z            = f_z(z,dist)
    neg_grad       = f_z/(sigma*(1-F_z(z,dist)))
    data_type      = rep("Right",n.points)
    parameter_type = rep('Negative Gradient',n.points)
    data           = data.frame(y.hat = y.hat,parameter=neg_grad,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="interval"){
    z_u           = (log(t.higher) - log(y.hat))/sigma
    z_l           = (log(t.lower) - log(y.hat))/sigma
    f_z_u         = f_z(z_u,dist)
    f_z_l         = f_z(z_l,dist)
    F_z_u         = F_z(z_u,dist)
    F_z_l         = F_z(z_l,dist)
    neg_grad      = -(f_z_u-f_z_l)/(sigma*(F_z_u-F_z_l))
    data_type     = rep("Interval",n.points)
    parameter_type = rep('Negative Gradient',n.points)
    data          = data.frame(y.hat = y.hat,parameter=neg_grad,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
}


hessian <- function(type="left",t.lower=NULL,t.higher=NULL,sigma=1,y.hat=1,dist='normal'){
  
  n.points      = length(y.hat)
  t.lower.col   = rep(t.lower,n.points)
  t.higher.col  = rep(t.higher,n.points)
  dist_type     = rep(dist,n.points)
  if(type=="uncensored"){
    z             = (log(t.lower) - log(y.hat))/sigma  
    f_z           = f_z(z,dist)
    grad_f_z      = grad_f_z(z,dist)
    hes_f_z       = hes_f_z(z,dist)
    hess           = -(f_z*hes_f_z - grad_f_z**2)/(sigma**2*f_z**2)
    data_type     = rep("Uncensored",n.points)
    parameter_type = rep('Hessian',n.points)
    data          = data.frame(y.hat = y.hat,parameter=hess,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="left"){
    z             = (log(t.higher) - log(y.hat))/sigma
    f_z           = f_z(z,dist)
    F_z           = F_z(z,dist)
    grad_f_z      = grad_f_z(z,dist)
    hess          = -(F_z*grad_f_z-f_z**2)/(sigma**2*F_z**2)
    data_type     = rep("Left",n.points)
    parameter_type = rep('Hessian',n.points)
    data          = data.frame(y.hat = y.hat,parameter=hess,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="right"){
    z             = (log(t.lower) - log(y.hat))/sigma
    f_z           = f_z(z,dist)
    F_z           = F_z(z,dist)
    grad_f_z      = grad_f_z(z,dist)
    hess          = -((1-F_z)*grad_f_z+f_z**2)/(sigma**2*(1-F_z)**2)
    data_type     = rep("Right",n.points)
    parameter_type = rep('Hessian',n.points)
    data          = data.frame(y.hat = y.hat,parameter=hess,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="interval"){
    z_u           = (log(t.higher) - log(y.hat))/sigma
    z_l           = (log(t.lower)  - log(y.hat))/sigma
    f_z_u         = f_z(z_u,dist)
    f_z_l         = f_z(z_l,dist)
    F_z_u         = F_z(z_u,dist)
    F_z_l         = F_z(z_l,dist)
    grad_f_z_u    = grad_f_z(z_u,dist)
    grad_f_z_l    = grad_f_z(z_l,dist) 
    hess          = ((F_z_u-F_z_l)*(grad_f_z_u+grad_f_z_l)-(f_z_u**2-f_z_l**2))/(sigma**2*(F_z_u-F_z_l)**2)
    data_type     = rep("Interval",n.points)
    parameter_type = rep('Hessian',n.points)
    data          = data.frame(y.hat = y.hat,parameter=hess,parameter_type = parameter_type,data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
}


n.points = 20
x.lim    = 15
distribution.list = list(gaussian=list(uncensored_l=loss(type="uncensored",t.lower=100,t.higher=100,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       left_l=loss(type="left",t.lower=-Inf,t.higher=20,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       right_l=loss(type="right",t.lower=60,t.higher=Inf,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       interval_l=loss(type="interval",t.lower=16,t.higher=200,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       uncensored_g=neg_grad(type="uncensored",t.lower=100,t.higher=100,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       left_g=neg_grad(type="left",t.lower=-Inf,t.higher=20,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       right_g=neg_grad(type="right",t.lower=60,t.higher=Inf,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       interval_g=neg_grad(type="interval",t.lower=16,t.higher=200,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       uncensored_h=hessian(type="uncensored",t.lower=100,t.higher=100,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       left_h=hessian(type="left",t.lower=-Inf,t.higher=20,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       right_h=hessian(type="right",t.lower=60,t.higher=Inf,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal'),
                                       interval_h=hessian(type="interval",t.lower=16,t.higher=200,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='normal')),
                         logistic=list(uncensored_l=loss(type="uncensored",t.lower=100,t.higher=100,sigma=sqrt(pi^2/6),y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       left_l=loss(type="left",t.lower=-Inf,t.higher=20,sigma=sqrt(pi^2/6),y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       right_l=loss(type="right",t.lower=60,t.higher=Inf,sigma=sqrt(pi^2/6),y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       interval_l=loss(type="interval",t.lower=16,t.higher=200,sigma=sqrt(pi^2/6),y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       uncensored_g=neg_grad(type="uncensored",t.lower=100,t.higher=100,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       left_g=neg_grad(type="left",t.lower=-Inf,t.higher=20,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       right_g=neg_grad(type="right",t.lower=60,t.higher=Inf,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       interval_g=neg_grad(type="interval",t.lower=16,t.higher=200,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       uncensored_h=hessian(type="uncensored",t.lower=100,t.higher=100,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       left_h=hessian(type="left",t.lower=-Inf,t.higher=20,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       right_h=hessian(type="right",t.lower=60,t.higher=Inf,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic'),
                                       interval_h=hessian(type="interval",t.lower=16,t.higher=200,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)),dist='logistic')))

data_complete_list <- list()
for(distribution in names(distribution.list)){
  loss.fun.list <- distribution.list[[distribution]]
  for(type in names(loss.fun.list)){
    data_complete_list[[paste(distribution, type)]] <- loss.fun.list[[type]]
  }
}

data_complete <- do.call(rbind, data_complete_list)
png("loss_grad_hess_aft.png", width = 800, height = 600)

p <- ggplot(data=data_complete) +
  geom_line(aes(x=y.hat,y=parameter,colour=dist_type),
            data=data_complete,size=1) + scale_x_continuous(trans='log2') +
  geom_point(aes(t.lower.col,y=0),data=data_complete) +
  geom_point(aes(t.higher.col,y=0),data=data_complete)+ theme(legend.position=c(0.1,0.9)) +
  ylab("loss function L_i(y_pred)")+
  xlab("predicted survival time y_pred in days (log_2 scale)") + facet_grid(parameter_type ~data_type ,scales="free") + 
  scale_color_discrete(name = "Distribution")
p
dev.off()

