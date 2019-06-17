set.seed(2)
library(ggplot2)
setwd('/Users/avinashbarnwal/Desktop/Personal/GSOC-2019/AFT/R')


z <- function(t.obs=t.obs,y.hat=y.hat,sigma=sigma){
  z= (log(t.obs)-log(y.hat))/sigma
  return(z)
}

f_z <- function(z=z,dist='logistic'){
  if(dist=='logistic'){
    f_z = e^z/(1+e^z)^2
  }
  if(dist=='normal'){
    f_z = dnorm(z)
  }
  return(f_z)
}

grad_f_z <- function(z=z,dist='logistic'){
  f_z      = f_z(z,dist)
  if(dist == 'logistic'){
    grad_f_z = f_z*(1-e^z)/(1+e^z)
  }
  if(dist == 'normal'){
    grad_f_z = -zf_z
  }
  return(grad_f_z)
}

hes_f_z <- function(z=z,dist='logistic'){
  f_z = f_z(z,dist)
  if(dist=='normal'){
    hes_f_z = -f(z) - z*grad_f_z(z,dist)    
  }
  if(dist=='logistic'){
    w = e^z
    hes_f_z = ((1-w)*grad_f_z(z,dist))/(1+w) - f_z/(1+w)
  }
}

F_z <- function(z,dist='logistic'){
  if(dist=='logistic'){
    F_z = e^z/(1+e^z)
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


neg_grad <- function(type="left",t.lower=NULL,t.higher=NULL,sigma=1,y.hat=1,dist='normal'){
  
  n.points      = length(y.hat)
  t.lower.col   = rep(t.lower,n.points)
  t.higher.col  = rep(t.higher,n.points)
  dist_type     = rep(dist,n.points)
  if(type=="uncensored"){
    z             = (log(t.lower) - log(y.hat))/sigma  
    f_z           = f_z(z,dist)
    neg_grad      = -grad_f_z(z,dist)/(sigma*f_z(z,dist))
    data_type     = rep("Uncensored",n.points)
    parameter_type = rep('Negative Gradient',n.points)
    data          = data.frame(y.hat = y.hat,parameter=neg_grad,parameter_type = 'Hessian',data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="left"){
    z              = (log(t.higher) - log(y.hat))/sigma
    f_z            = f_z(z,dist)
    neg_grad       = -f_z/(sigma*F_z(z,dist))
    data_type      = rep("Left",n.points)
    parameter_type = rep('Negative Gradient',n.points)
    data           = data.frame(y.hat = y.hat,parameter=neg_grad,parameter_type = 'Hessian',data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="right"){
    z              = (log(t.lower) - log(y.hat))/sigma
    f_z            = f_z(z,dist)
    neg_grad       = f_z/(sigma*(1-F_z(z,dist)))
    data_type      = rep("Right",n.points)
    parameter_type = rep('Negative Gradient',n.points)
    data           = data.frame(y.hat = y.hat,parameter=neg_grad,parameter_type = 'Hessian',data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
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
    data_type     = rep("Right",n.points)
    parameter_type = rep('Negative Gradient',n.points)
    data          = data.frame(y.hat = y.hat,parameter=neg_grad,parameter_type = 'Hessian',data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
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
    hess           = -grad_f_z(z,dist)/(sigma*f_z(z,dist))
    data_type     = rep("Uncensored",n.points)
    parameter_type = rep('Hessian',n.points)
    data          = data.frame(y.hat = y.hat,parameter=hess,parameter_type = 'Hessian',data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
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
    data          = data.frame(y.hat = y.hat,parameter=hess,parameter_type = 'Hessian',data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
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
    data          = data.frame(y.hat = y.hat,parameter=hess,parameter_type = 'Hessian',data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
  if(type=="interval"){
    z_u           = (log(t.higher) - log(y.hat))/sigma
    z_l           = (log(t.lower) - log(y.hat))/sigma
    f_z_u         = f_z(z_u,dist)
    f_z_l         = f_z(z_l,dist)
    F_z_u         = F_z(z_u,dist)
    F_z_l         = F_z(z_l,dist)
    grad_f_z_u    = grad_f_z(z_u,dist)
    grad_f_z_l    = grad_f_z(z_l,dist) 
    hess          = ((F_z_u-F_z_l)*(grad_f_z_u-grad_f_z_l)-(f_z_u**2-f_z_l**2))/(sigma**2*(F_z_u-F_z_l)**2)
    data_type     = rep("Right",n.points)
    parameter_type = rep('Hessian',n.points)
    data          = data.frame(y.hat = y.hat,parameter=hess,parameter_type = 'Hessian',data_type = data_type,dist_type = dist_type,t.lower.col=t.lower.col,t.higher.col=t.higher.col)
    return(data)
  }
}

