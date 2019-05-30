from libc cimport math
import  numpy  as np
cimport numpy as cnp
from scipy.stats import norm

cnp.import_array()

@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)

def aft_negative_gradient(cnp.npy_double[:] y_lower,
                          cnp.npy_double[:] y_higher,
                          cnp.npy_double[:] y_pred,
                          char* dist = dist,
                          cnp.npy_double sigma):
    
    cdef cnp.npy_intp n_samples = len(y_lower)
    cdef cnp.ndarray[cnp.npy_double, ndim=1] gradient = cnp.PyArray_EMPTY(1, &n_samples, cnp.NPY_DOUBLE, 0)
    cdef cnp.ndarray[cnp.npy_string, ndim=1] event = cnp.PyArray_EMPTY(1, &n_samples, cnp.NPY_STRING, 0)
    
    for i in range(n_samples):
        if y_lower==y_higher:
            event[i] = 'uncensored'
        elif y_lower != -float('inf') and y_higher != float('inf'):
            event[i] = 'interval'
        elif y_lower == -float('inf'):
            event[i] = 'left'
        else:
            event[i] = 'right'
            
    cdef cnp.npy_double[:] exp_pred = np.exp(y_pred)
    
    with nogil:
        if dist == 'normal':
            for i in range(n_samples):
                if event[i] == 'uncensored':
                    gradient[i] = (-sigma**2 + math.log(y_lower[i]/exp_pred[i]))/(sigma**2*exp_pred[i])
                elif event[i] == 'left':
                    gradient[i] = -math.sqrt(2/math.pi)*np.exp(math.log(y_higher[i]/exp_pred[i])**2/(2*sigma**2))/\
                    (sigma*exp_pred[i]*(math.erf(math.log(y_higher[i]/exp_pred[i])/((math.sqrt(2)*sigma)))))
                elif event[i] == 'right':
                    gradient[i] = np.exp(math.log(y_lower[i]/exp_pred[i])**2/(2*sigma**2))/\
                    (math.sqrt(2*math.pi)*sigma*exp_pred[i]*(0.5*(-1-math.erf(math.log(y_lower[i]/exp_pred[i])/((math.sqrt(2)*sigma))))+1))
                else :
                    gradient[i] = ((math.sqrt(2/math.pi)*np.exp(math.log(y_lower[i]/exp_pred[i])**2/(2*sigma**2)))-
                                  (math.sqrt(2/math.pi)*np.exp(math.log(y_higher[i]/exp_pred[i])**2/(2*sigma**2))))/\
                    (math.erf(math.log(y_higher[i]/exp_pred[i])/((math.sqrt(2)*sigma)))-math.erf(math.log(y_lower[i]/exp_pred[i])/((math.sqrt(2)*sigma))))
                    
        elif dist == 'logistic':
            for i in range(n_samples):
                if event[i] == 'uncensored':
                    gradient[i] = 
                elif event[i] == 'left':
                    gradient[i] = -math.sqrt(2/math.pi)*np.exp(math.log(y_higher[i]/exp_pred[i])**2/(2*sigma**2))/\
                    (sigma*exp_pred[i]*(math.erf(math.log(y_higher[i]/exp_pred[i])/((math.sqrt(2)*sigma)))))
                elif event[i] == 'right':
                    gradient[i] = np.exp(math.log(y_lower[i]/exp_pred[i])**2/(2*sigma**2))/\
                    (math.sqrt(2*math.pi)*sigma*exp_pred[i]*(0.5*(-1-math.erf(math.log(y_lower[i]/exp_pred[i])/((math.sqrt(2)*sigma))))+1))
                else :
                    gradient[i] = ((math.sqrt(2/math.pi)*np.exp(math.log(y_lower[i]/exp_pred[i])**2/(2*sigma**2)))-
                                  (math.sqrt(2/math.pi)*np.exp(math.log(y_higher[i]/exp_pred[i])**2/(2*sigma**2))))/\
                    (math.erf(math.log(y_higher[i]/exp_pred[i])/((math.sqrt(2)*sigma)))-math.erf(math.log(y_lower[i]/exp_pred[i])/((math.sqrt(2)*sigma))))

    return gradient

@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)

def aft_loss(cnp.npy_double[:] y_lower,
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