#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double neg_grad_right(double y_lower,double y_higher,double y_pred,double sigma,std::string dist)
{
	double pdf;
	double z;
	double cdf;
	double neg_grad;
	z    = (std::log(y_lower)-y_pred)/sigma;
  if(dist=="normal"){
  		pdf  = dnorm(z,0,1);
  		cdf  = pnorm(z,0,1);
  }
  else{
  		pdf  = dlogis(z,0,1);
  		cdf  = plogis(z,0,1); 
  }

	neg_grad = pdf/(sigma*std::max(0.000005,1-cdf));

  return neg_grad;
}
