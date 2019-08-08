#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"

extern "C" double loss_uncensored(double y_lower,double y_higher,double y_pred,double sigma,char* dist)
{
  double z;
  double pdf;
  double cost;
  char*  given_dist = "normal";
  z       = (std::log(y_lower)-y_pred)/sigma;
  if(strcmp(dist, given_dist) == 0){
  	pdf = dnorm(z,0,1);
  }else{
  	pdf = dlogis(z,0,1);
  }
  cost = -std::log(pdf/(sigma*y_lower));
  return cost;
}
