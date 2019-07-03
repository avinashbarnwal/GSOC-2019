library(ggplot2)
library(gridExtra)
setwd('/Users/avinashbarnwal/Desktop/Personal/GSOC-2019/AFT/R')
#http://home.iitk.ac.in/~kundu/paper146.pdf
#https://www.mathworks.com/matlabcentral/answers/428624-cdf-for-loglogistic-distribution
set.seed(2)

loss   <- function(n.obs = 100, y.obs = 0.5, y.hat=1){
  loss <- -n.obs*y.obs*y.hat + n.obs*log(1+exp(y.hat)) 
  return(loss)
}

neg_gradient   <- function(n.obs = 100, y.obs = 0.5, y.hat=1){
  neg_gradient <- n.obs*y.obs - n.obs*(exp(y.hat)/(1+exp(y.hat)))
  return(neg_gradient)
}

hessian   <- function(n.obs = 100, y.obs = 0.5, y.hat=1){
  hessian <- n.obs*(exp(y.hat)/(1+exp(y.hat))**2)
  return(hessian)
}

binomial_properties <- function(n.obs = 100, y.obs = 0.5, y.hat=1,type="loss"){
  if(type=="loss"){
    loss         = loss(n.obs,y.obs,y.hat)
    n.points     = length(y.hat)
    type_series  = rep(type,n.points)
    data         = data.frame(y.hat = y.hat, cost=loss, data_type = type_series)
    return(data)
  }
  if(type=="neg_gradient"){
    neg_gradient = neg_gradient(n.obs,y.obs,y.hat)
    n.points     = length(y.hat)
    type_series  = rep(type,n.points)
    data         = data.frame(y.hat = y.hat, cost=neg_gradient, data_type = type_series)
    return(data)
  }
  if(type=="hessian"){
    hessian      = hessian(n.obs,y.obs,y.hat)
    n.points     = length(y.hat)
    type_series  = rep(type,n.points)
    data         = data.frame(y.hat = y.hat, cost=hessian, data_type = type_series)
    return(data)
  }
}

n.points  = 20
x.lim     = 50

distribution.list  = list(loss         = binomial_properties(type="loss", n.obs = 100, y.obs = 0.5, y.hat=seq(-50,x.lim,length=n.points)),
                          neg_gradient = binomial_properties(type="neg_gradient", n.obs = 100, y.obs = 0.5, y.hat=seq(-50,x.lim,length=n.points)),
                          hessian      = binomial_properties(type="hessian", n.obs = 100, y.obs = 0.5, y.hat=seq(-50,x.lim,length=n.points))
                          )

data_complete_list = list()

for(distribution in names(distribution.list)){
  data_complete_list[[distribution]] = distribution.list[[distribution]]
}

data_complete = do.call(rbind, data_complete_list)



#data_loss         = data_complete[which(data_complete['data_type']=='loss'),]
#data_neg_gradient = data_complete[which(data_complete['data_type']=='neg_gradient'),]
#data_hessian      = data_complete[which(data_complete['data_type']=='hessian'),]

#par(mfrow=c(3,1))
#p1 = ggplot(data = data_loss) +
#     geom_line(aes(x = y.hat,y=cost),
#                   data = data_loss,size=1) +
#     ylab("Loss") +
#     xlab("predicted survival time y_pred in days")

#p2 = ggplot(data = data_neg_gradient) +
#     geom_line(aes(x = y.hat, y=cost),data = data_neg_gradient,size=1) +
#     theme(legend.position=c(0.1,0.8)) +
#     ylab("Negative Gradient") +
#     xlab("predicted survival time y_pred in days")

#p3 = ggplot(data = data_hessian) +
#     geom_line(aes(x = y.hat,y=cost),data = data_hessian,size=1) +
#     ylab("Hessian") +
#     xlab("predicted survival time y_pred in days")
#scale_color_discrete(name = "Distribution")
#dev.off()
#grid.arrange(p1, p2, p3, nrow = 1)


png("binomial_loss.png", width = 800, height = 600)
p <- ggplot(data=data_complete) +
  geom_line(aes(x=y.hat,y=cost,colour=data_type),
            data=data_complete,size=1) +
  ylab("loss function L_i(y_pred)")    +
  xlab("predicted survival time y_pred in days") + facet_grid(data_type ~ .,scales="free") + 
  scale_color_discrete(name = "Distribution")
p
dev.off()
