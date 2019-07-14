#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double hess_logis(double x, double mu, double sd)
{
	double pdf;
	double z;
	double hess;
	double w;
	pdf     = dlogis(x,mu,sd);
	z       = (x-mu)/sd;
	w       = std::pow(std::exp(1),z);
    hess    = pdf*(std::pow(w,2)-4*w+1)/std::pow((1+w),2);
    return hess;
}
