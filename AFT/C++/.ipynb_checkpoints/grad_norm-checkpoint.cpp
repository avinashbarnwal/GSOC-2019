#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double _grad_norm(double x, double mu, double sd)
{
	double pdf;
	double z;
	double grad;
	pdf  = dnorm(x,mu,sd);
	z = (x-mu)/sd;
	grad = -1*z*pdf;
    return grad;
}
