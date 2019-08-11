#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double hessian_left(double y_lower,double y_higher,double y_pred,double sigma,char* dist)
{
	double pdf;
  double cdf;
	double z;
	double grad;
	double hess;
  char* given_dist1 =  "normal";
  char* given_dist2 =  "logistic";
  char* given_dist3 =  "extreme";

  z    = (std::log(y_higher)-y_pred)/sigma;
  if(strcmp(dist, given_dist1) == 0){
  		pdf       = dnorm(z,0,1);
      cdf       = pnorm(z,0,1);
  		grad      = grad_norm(z,0,1);
  }
  else if(strcmp(dist, given_dist2) == 0){
  		pdf  = dlogis(z,0,1);
      cdf  = plogis(z,0,1);
  		grad = grad_logis(z,0,1);
  }
  else if(strcmp(dist, given_dist3) == 0){
      pdf  = dextreme(z,0,1);
      cdf  = pextreme(z,0,1);
      grad = grad_extreme(z,0,1);
  }
	hess = -(cdf*grad - std::pow(pdf,2))/(std::pow(sigma,2)*std::pow(cdf,2));
  return hess;
}


