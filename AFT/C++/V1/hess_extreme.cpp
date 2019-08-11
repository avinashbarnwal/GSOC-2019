#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double hess_extreme(double x, double mu, double sd)
{
	double pdf;
	double z;
	double hess;
	double w;
	pdf     = dextreme(x,mu,sd);
	z       = (x-mu)/sd;
	w       = std::exp(z);
	hess    = (std::pow(w,2)-3*w+1)*pdf;
    return hess;
}
