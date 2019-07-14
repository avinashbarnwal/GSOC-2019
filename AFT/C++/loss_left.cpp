#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"

extern "C" double loss_uncensored(double y_lower,double y_higher,double y_pred,double sigma,std::string dist)
{
  double z;
  double cdf;
  double cost;	
  z    = (std::log(y_higher)-y_pred)/sigma;
  if(dist=="normal"){
  	cdf = pnorm(z,0,1);
  }
  else{
  	cdf = plogis(z,0,1);
  }
  cost = -std::log(cdf);  
  return cost;
}
