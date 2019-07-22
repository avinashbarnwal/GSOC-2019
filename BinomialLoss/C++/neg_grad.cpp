#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double neg_grad(double n_obs,double y_obs,double y_hat)
{
	double neg_grad;
  neg_grad  = y_obs - n_obs*(std::exp(y_hat)/(1+std::exp(y_hat)));
  return neg_grad;
}
