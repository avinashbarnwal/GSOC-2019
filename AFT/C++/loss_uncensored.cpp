#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"

extern "C" double loss_uncensored(double y_lower,double y_higher,double y_pred,double sigma,std::string dist)
{
  double z;
  double pdf;
  double cost;	
  z    = (std::log(y_lower)-y_pred)/sigma;
  if(dist=="normal"){
  	pdf = dnorm(z,0,1);
  }
  else{
  	pdf = dlogis(z,0,1);
  }
  cost = -std::log(std::max(0.00005,pdf/(sigma*y_lower)));
  return cost;
}
