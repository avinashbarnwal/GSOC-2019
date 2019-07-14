#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"

extern "C" double loss_uncensored(double y_lower,double y_higher,double y_pred,double sigma,std::string dist)
{
  double z_u;
  double z_l;
  double cdf_u;
  double cdf_l;
  double cost;
  z_u   = (std::log(y_higher) - y_pred)/sigma;
  z_l   = (std::log(y_lower) - y_pred)/sigma;
  if(dist=="normal"){
  	cdf_u = pnorm(z_u,0,1);
    cdf_l = pnorm(z_l,0,1);
  }
  else{
  	cdf_u = plogis(z_u,0,1);
    cdf_l = plogis(z_l,0,1);
  }
  cost = -std::log(std::max(0.00005,cdf_u - cdf_l)); 
  return cost;
}
