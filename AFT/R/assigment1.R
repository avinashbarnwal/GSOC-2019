library(ggplot2)
setwd('/Users/avinashbarnwal/Desktop/Personal/GSOC-2019/AFT/R')
#http://home.iitk.ac.in/~kundu/paper146.pdf

set.seed(2)
n.points = 15

loss_lognormal <- function(type="left",t.lower=NULL,t.higher=NULL,sigma=1,y.hat=1){
  
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


loss_loglogistic <- function(type="left",t.lower=NULL,t.higher=NULL,sigma=sqrt(pi^2/6),y.hat=1){
  
  n.points      = length(y.hat)
  t.lower.col   = rep(t.lower,n.points)
  t.higher.col  = rep(t.higher,n.points)
  dist_type     = rep("Logistic",n.points)
  
  if(type=="uncensored"){
    cost = -log((1/(sigma*t.lower))*(exp(log(t.lower/y.hat)/sigma)/(1+exp(log(t.lower/y.hat)/sigma))**2))
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
n.points = 20
x.lim    = 15
distribution.list <- list(gaussian=list(uncensored=loss_lognormal(type="uncensored",t.lower=100,t.higher=100,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points))),
                                        left=loss_lognormal(type="left",t.lower=-Inf,t.higher=20,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points))),
                                        right=loss_lognormal(type="right",t.lower=60,t.higher=Inf,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points))),
                                        interval=loss_lognormal(type="interval",t.lower=16,t.higher=200,sigma=1,y.hat=2**(seq(1,x.lim,length=n.points)))),
                          logistic=list(uncensored=loss_loglogistic(type="uncensored",t.lower=100,t.higher=100,sigma=sqrt(pi^2/6),y.hat=2**(seq(1,x.lim,length=n.points))),
                                        left=loss_loglogistic(type="left",t.lower=-Inf,t.higher=20,sigma=sqrt(pi^2/6),y.hat=2**(seq(1,x.lim,length=n.points))),
                                        right=loss_loglogistic(type="right",t.lower=60,t.higher=Inf,sigma=sqrt(pi^2/6),y.hat=2**(seq(1,x.lim,length=n.points))),
                                        interval=loss_loglogistic(type="interval",t.lower=16,t.higher=200,sigma=sqrt(pi^2/6),y.hat=2**(seq(1,x.lim,length=n.points)))))

data_complete_list <- list()
for(distribution in names(distribution.list)){
  loss.fun.list <- distribution.list[[distribution]]
  for(type in names(loss.fun.list)){
    data_complete_list[[paste(distribution, type)]] <- loss.fun.list[[type]]
  }
}

data_complete <- do.call(rbind, data_complete_list)

png("loss_aft.png", width = 800, height = 600)

p <- ggplot(data=data_complete) +
     geom_line(aes(x=y.hat,y=cost,colour=dist_type),
              data=data_complete,size=1) + scale_x_continuous(trans='log2') +
     geom_point(aes(t.lower.col,y=0),data=data_complete) +
     geom_point(aes(t.higher.col,y=0),data=data_complete)+ theme(legend.position=c(0.1,0.8))+
     ylab("loss function L_i(y_pred)")+
     xlab("predicted survival time y_pred in days (log_2 scale)") + facet_grid(. ~ data_type,scales="free") + 
     scale_color_discrete(name = "Distribution")
p
dev.off()


