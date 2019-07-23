#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double hessian_right(double y_lower,double y_higher,double y_pred,double sigma,char* dist)
{
	double pdf;
  double cdf;
	double z;
	double grad;
	double hess;
  char* given_dist =  "normal";
  z    = (std::log(y_lower)-y_pred)/sigma;
  if(strcmp(dist, given_dist) == 0){
  		pdf       = dnorm(z,0,1);
      cdf       = pnorm(z,0,1);
  		grad      = grad_norm(z,0,1);
  } else{
  		pdf  = dlogis(z,0,1);
      cdf  = plogis(z,0,1);
  		grad = grad_logis(z,0,1);
  }
	hess = ((1-cdf)*grad + std::pow(pdf,2))/(std::pow(sigma,2)*std::pow(1-cdf,2));
  return hess;
}
