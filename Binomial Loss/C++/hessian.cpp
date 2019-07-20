#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double hessian(double n_obs,double y_obs,double y_hat)
{
	double hess;
	hess = n_obs*(std::exp(y_hat)/std::pow((1+std::exp(y_hat)),2));
  return hess;
}
