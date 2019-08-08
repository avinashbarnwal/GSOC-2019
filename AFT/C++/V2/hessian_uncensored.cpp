#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double hessian_uncensored(double y_lower,double y_higher,double y_pred,double sigma,char* dist)
{
	double pdf;
	double z;
	double grad;
	double hess;
  double hess_dist;
  char* given_dist =  "normal";
  z    = (std::log(y_lower)-y_pred)/sigma;
  if(strcmp(dist, given_dist) == 0){
  		pdf       = dnorm(z,0,1);
  		grad      = grad_norm(z,0,1);
      hess_dist = hess_norm(z,0,1);
  	}
  	else{
  		pdf  = dlogis(z,0,1);
  		grad = grad_logis(z,0,1);
      hess_dist = hess_logis(z,0,1);
  	}
	hess = -(pdf*hess_dist - std::pow(grad,2))/(std::pow(sigma,2)*std::pow(pdf,2));
  return hess;
}
