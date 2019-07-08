library(ggplot2)
library(gridExtra)
setwd('/Users/avinashbarnwal/Desktop/Personal/GSOC-2019/AFT/R')
#http://home.iitk.ac.in/~kundu/paper146.pdf
#https://www.mathworks.com/matlabcentral/answers/428624-cdf-for-loglogistic-distribution
set.seed(2)

loss   <- function(n.obs = 100, y.obs = 50, y.hat=1){
  loss <- -y.obs*y.hat + n.obs*log(1+exp(y.hat)) 
  return(loss)
}

neg_gradient   <- function(n.obs = 100, y.obs = 50, y.hat=1){
  neg_gradient <- y.obs - n.obs*(exp(y.hat)/(1+exp(y.hat)))
  return(neg_gradient)
}

hessian   <- function(n.obs = 100, y.obs = 50, y.hat=1){
  hessian <- n.obs*(exp(y.hat)/(1+exp(y.hat))**2)
  return(hessian)
}

binomial_properties <- function(n.obs = 100, y.obs = 50, y.hat=1,type="loss"){
  if(type=="loss"){
    loss         = loss(n.obs,y.obs,y.hat)
    n.points     = length(y.hat)
    type_series  = rep(type,n.points)
    y.obs.tag    = paste('y.obs=',y.obs,sep="")
    y.obs.col    = rep(y.obs.tag,n.points)
    data         = data.frame(y.hat = y.hat, cost=loss, data_type = type_series,y_obs = y.obs.col)
    return(data)
  }
  if(type=="neg_gradient"){
    neg_gradient = neg_gradient(n.obs,y.obs,y.hat)
    n.points     = length(y.hat)
    type_series  = rep(type,n.points)
    y.obs.tag    = paste('y.obs=',y.obs,sep="")
    y.obs.col    = rep(y.obs.tag,n.points)
    data         = data.frame(y.hat = y.hat, cost=neg_gradient, data_type = type_series,y_obs = y.obs.col)
    return(data)
  }
  if(type=="hessian"){
    hessian      = hessian(n.obs,y.obs,y.hat)
    n.points     = length(y.hat)
    type_series  = rep(type,n.points)
    y.obs.tag    = paste('y.obs=',y.obs,sep="")
    y.obs.col    = rep(y.obs.tag,n.points)
    data         = data.frame(y.hat = y.hat, cost=hessian, data_type = type_series,y_obs = y.obs.col)
    return(data)
  }
}

n.points  = 200
x.lim     = 50

distribution.list  = list(loss_50            = binomial_properties(type="loss", n.obs = 100, y.obs = 50, y.hat=seq(-50,x.lim,length=n.points)),
                          loss_0             = binomial_properties(type="loss", n.obs = 100, y.obs  = 0, y.hat=seq(-50,x.lim,length=n.points)),
                          loss_20            = binomial_properties(type="loss", n.obs = 100, y.obs = 20, y.hat=seq(-50,x.lim,length=n.points)),
                          neg_gradient_50    = binomial_properties(type="neg_gradient", n.obs = 100, y.obs = 50, y.hat=seq(-50,x.lim,length=n.points)),
                          neg_gradient_0     = binomial_properties(type="neg_gradient", n.obs = 100, y.obs = 0, y.hat=seq(-50,x.lim,length=n.points)),
                          neg_gradient_20    = binomial_properties(type="neg_gradient", n.obs = 100, y.obs = 20, y.hat=seq(-50,x.lim,length=n.points)),
                          hessian_50         = binomial_properties(type="hessian", n.obs = 100, y.obs = 50, y.hat=seq(-50,x.lim,length=n.points)),
                          hessian_0          = binomial_properties(type="hessian", n.obs = 100, y.obs = 0, y.hat=seq(-50,x.lim,length=n.points)),
                          hessian_20         = binomial_properties(type="hessian", n.obs = 100, y.obs = 20, y.hat=seq(-50,x.lim,length=n.points))
                          )

data_complete_list = list()

for(distribution in names(distribution.list)){
  data_complete_list[[distribution]] = distribution.list[[distribution]]
}

data_complete = do.call(rbind, data_complete_list)
png("binomial_loss.png", width = 800, height = 600)
p = ggplot(data=data_complete) +
    geom_line(aes(x=y.hat,y=cost),
              data=data_complete,size=1) +
    ylab("Metrics performance")    +
    xlab("eta") + facet_grid(data_type ~ y_obs,scales="free") + 
    scale_color_discrete(name = "Distribution")
p
dev.off()
