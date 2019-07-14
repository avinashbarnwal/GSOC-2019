#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double hess_norm(double x, double mu, double sd)
{
	double pdf;
	double z;
	double hess;
	pdf     = dnorm(x,mu,sd);
	z       = (x-mu)/sd;
	hess = (std::pow(z,2)-1)*pdf;
    return hess;
}
