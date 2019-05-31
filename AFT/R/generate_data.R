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
