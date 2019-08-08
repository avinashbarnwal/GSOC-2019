#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"

extern "C" double loss_left(double y_lower,double y_higher,double y_pred,double sigma,char* dist)
{
  double z;
  double cdf;
  double cost;	
  char*  given_dist = "normal";
  z    = (std::log(y_higher)-y_pred)/sigma;
  if(strcmp(dist, given_dist) == 0){
  	cdf = pnorm(z,0,1);
  }
  else{
  	cdf = plogis(z,0,1);
  }
  cost = -std::log(cdf);  
  return cost;
}
