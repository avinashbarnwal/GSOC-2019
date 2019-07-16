
#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double neg_grad_uncensored(double y_lower,double y_higher,double y_pred,double sigma,std::string dist)
{
	double pdf;
	double z;
	double grad;
	double neg_grad;
	z    = (std::log(y_lower)-y_pred)/sigma;
  	if(dist=="normal"){
  		pdf  = dnorm(z,0,1);
  		grad = grad_norm(z,0,1);
  	}
  	else{
  		pdf  = dlogis(z,0,1);
  		grad = grad_logis(z,0,1); 
  	}

	neg_grad = -grad/(sigma*pdf);
    return neg_grad;
}
