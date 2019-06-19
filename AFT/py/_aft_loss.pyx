from libc cimport math
import  numpy  as np
cimport numpy as cnp
from scipy.stats import norm

cnp.import_array()

@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)

def getEventType(cnp.npy_double y_lower, cnp.npy_double y_higher):
    if y_lower==y_higher:
        return 'uncensored'
    elif y_lower != -float('inf') and y_higher != float('inf'):
        return 'interval'
    elif y_lower == -float('inf'):
        return 'left'
    else:
        return 'right'

def z(cnp.npy_double y, cnp.npy_double y_pred, cnp.npy_double sigma):
    z = (math.log(y)-y_pred)/sigma
    return z

def f_z(cnp.npy_double z, char* dist = 'logistic'):
    if dist == 'logistic':
        f_z = math.exp(1)**z/(1+math.exp(1)**z)**2
    if dist == 'normal':
        f_z = norm.pdf(z)
    return f_z

def grad_f_z(cnp.npy_double z, char* dist = 'logistic'):
    f_z = f_z(z,dist)
    if dist == 'logistic':
        grad_f_z = f_z*(1-math.exp(1)**z)/(1+math.exp(1)**z)
    if dist == 'normal':
        grad_f_z = -z*f_z
    return grad_f_z


def hes_f_z(cnp.npy_double z, char* dist = 'logistic'):
    f_z = f_z(z,dist)
    if dist == 'logistic':
        w       = math.exp(1)**z
        hes_f_z = f_z*(w**2-4*w+1)/(1+w)**2 
    if dist == 'normal':
        hes_f_z = (z**2-1)*f_z  
    return hes_f_z

def F_z(cnp.npy_double z, char* dist = 'logistic'):
    if dist=='logistic':
        F_z = math.exp(1)**z/(1+math.exp(1)**z)
    if dist=='normal':
        F_z = norm.cdf(z)
    return F_z

def grad_F_z(z,dist):
    return f_z(z,dist)
    
def f_y(cnp.npy_double z, cnp.npy_double y, cnp.npy_double sigma, char* dist = 'logistic'):
    f_y = f_z(z,dist)/(y*sigma)
    return f_y

def grad_f_y(cnp.npy_double z, cnp.npy_double y, cnp.npy_double sigma, char* dist = 'logistic'):
    grad_f_y = -grad_f_z(z,dist)/(sigma**2*y)
    return f_y

def _neg_grad(cnp.npy_double y_lower, cnp.npy_double y_higher, cnp.npy_double y_pred, cnp.npy_double sigma, char* type = 'left', char* dist = 'normal'):
    if type=='uncensored':
        z   = (math.log(y_lower)-y_pred)/sigma
        f_z = f_z(z,dist)
        _neg_grad = -grad_f_z(z,dist)/(sigma*f_z)
        return _neg_grad
    if type=='left':
        z   = (math.log(y_higher)-y_pred)/sigma
        f_z = f_z(z,dist)
        _neg_grad = -f_z/(sigma*F_z(z,dist))
        return _neg_grad
    if type=='right':
        z   = (math.log(y_lower)-y_pred)/sigma
        f_z = f_z(z,dist)
        _neg_grad = f_z/(sigma*(1-F_z(z,dist)))
        return _neg_grad
    if type=='interval':
        z_u           = (math.log(y_higher) - y_pred)/sigma
        z_l           = (math.log(y_lower) - y_pred)/sigma
        f_z_u         = f_z(z_u,dist)
        f_z_l         = f_z(z_l,dist)
        F_z_u         = F_z(z_u,dist)
        F_z_l         = F_z(z_l,dist)
        _neg_grad     = -(f_z_u-f_z_l)/(sigma*(F_z_u-F_z_l))
        return _neg_grad
    
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
    
    
def _loss(cnp.npy_double y_lower, cnp.npy_double y_higher, cnp.npy_double y_pred, cnp.npy_double sigma, char* type = 'left', char* dist = 'normal'):
    if type=='uncensored':
        z    = (math.log(y_lower)-y_pred)/sigma
        f_z  = f_z(z,dist)
        cost = -math.log(f_z/(sigma*y_lower))
        return cost
    if type=='left':
        z    = (math.log(y_higher)-y_pred)/sigma
        F_z  = F_z(z,dist)
        cost = -math.log(F_z)
        return cost
    if type=='right':
        z   = (math.log(y_lower)-y_pred)/sigma
        F_z = F_z(z,dist)
        cost= -math.log(1-F_z)
        return cost
    if type=='interval':
        z_u   = (math.log(y_higher) - y_pred)/sigma
        z_l   = (math.log(y_lower) - y_pred)/sigma
        f_z_u = f_z(z_u,dist)
        f_z_l = f_z(z_l,dist)
        F_z_u = F_z(z_u,dist)
        F_z_l = F_z(z_l,dist)
        cost  = -math.log(F_z_u - F_z_l)
        return cost
    
    
    
    
def negative_gradient(cnp.npy_double[:] y_lower,
                      cnp.npy_double[:] y_higher,
                      cnp.npy_double[:] y_pred,
                      char* dist = dist,
                      cnp.npy_double sigma):
    
    # Notation Convention
    # eta = Xb
    # This is original Equation
    # log(y_pred) = Xb
    # Here
    # eta = y_pred
    # Actual Predicted = exp^(Xb)
    
    
    cdef cnp.npy_intp n = len(y_lower)
    cdef cnp.ndarray[cnp.npy_double, ndim=1] gradient   = cnp.PyArray_EMPTY(1, &n_samples, cnp.NPY_DOUBLE, 0)
    cdef cnp.ndarray[cnp.npy_string, ndim=1] event      = cnp.PyArray_EMPTY(1, &n_samples, cnp.NPY_STRING, 0)
    cdef cnp.ndarray[cnp.npy_double, ndim=1] sigma_rep  = cnp.repeat(sigma,n)
    cdef cnp.ndarray[cnp.npy_double, ndim=1] dist_rep   = cnp.repeat(dist,n)
    
    event     = list(map(getEventType,zip(y_lower,y_higher)))
    neg_grad  = map(_neg_grad,zip(y_lower,y_higher,y_pred,sigma_rep,event,dist_rep))
    
    return neg_grad

@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)

def loss(cnp.npy_double[:] y_lower,
             cnp.npy_double[:] y_higher,
             cnp.npy_double[:] y_pred,
             char* dist = dist,
             cnp.npy_double sigma):
    
    cdef cnp.npy_intp n_samples = len(y_lower)
    cdef cnp.ndarray[cnp.npy_string, ndim=1] event = cnp.PyArray_EMPTY(1, &n_samples, cnp.NPY_STRING, 0)
    cdef cnp.npy_double loss = 0
    cdef cnp.npy_double[:] exp_pred = np.exp(y_pred)
    
    
    for i in range(n_samples):
        if y_lower==y_higher:
            event[i] = 'uncensored'
        elif y_lower != -float('inf') and y_higher != float('inf'):
            event[i] = 'interval'
        elif y_lower == -float('inf'):
            event[i] = 'left'
        else:
            event[i] = 'right'


    with nogil:
        if dist == 'normal':
            for i in range(n_samples):
                if event[i] == 'uncensored':
                    loss += math.log(1/(y_pred[i]*sigma*math.sqrt(2*3.14))*
                                     math.exp(math.log(y_lower[i]/exp_pred[i])/(-2*sigma*sigma)))
                elif event[i] == 'interval':
                    loss += math.log(norm.cdf(math.log(y_higher[i]/exp_pred[i])/(sigma)) -
                                     norm.cdf(math.log(y_lower[i]/exp_pred[i])/(sigma)))
                elif event[i] == 'left':
                    loss += math.log(norm.cdf(math.log(y_higher[i]/exp_pred[i])/sigma))
                else:
                    loss += math.log(1-norm.cdf(math.log(y_lower[i]/exp_pred[i])/sigma))
                    
                    
        elif dist == 'logistic':
            if event[i] == 'uncensored':
                loss += math.log((1/(sigma*exp_pred[i]))*
                            math.exp(math.log(y_lower[i]/exp_pred[i]))/(1+math.exp(math.log(y_lower[i]/exp_pred[i])))**2)
            elif event[i] == 'interval':
                loss += math.log(math.exp(math.log(y_higher[i]/exp_pred[i]))/(1+math.exp(math.log(y_higher[i]/exp_pred[i]))) -                             math.exp(math.log(y_lower[i]/exp_pred[i]))/(1+math.exp(math.log(y_lower[i]/exp_pred[i]))))
                elif event[i] == 'left':
                    loss += math.log(math.exp(math.log(y_higher[i]/exp_pred[i]))/(1+math.exp(math.log(y_higher[i]/exp_pred[i]))))
                else:
                    loss += math.log(1- math.exp(math.log(y_lower[i]/exp_pred[i]))/(1+math.exp(math.log(y_lower[i]/exp_pred[i]))))

    return -loss