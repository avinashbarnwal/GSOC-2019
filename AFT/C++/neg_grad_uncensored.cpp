
#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double neg_grad_uncensored(double y_lower,double y_higher,double y_pred,double sigma,char* dist)
{
	double pdf;
	double z;
	double grad;
	double neg_grad;
  char* given_dist =  "normal";
  z    = (std::log(y_lower)-y_pred)/sigma;
  if(strcmp(dist, given_dist) == 0){
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
