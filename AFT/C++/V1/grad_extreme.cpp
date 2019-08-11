#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double grad_extreme(double x, double mu, double sd)
{
	double pdf;
	double w;
	double z;
	double grad;
	pdf  = dnorm(x,mu,sd);
	z = (x-mu)/sd;
	w    = std::exp(z);
    grad = (1-w)*pdf;
    return grad;
}
