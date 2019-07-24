#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"

extern "C" double loss(double n_obs,double y_obs,double y_hat)
{
  double cost;
  cost = y_obs*std::log(1+std::exp(-y_hat)) + (n_obs-y_obs)*std::log(1+std::exp(y_hat));
  return cost;
}
