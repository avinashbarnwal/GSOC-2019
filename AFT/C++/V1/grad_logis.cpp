
#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double grad_logis(double x, double mu, double sd)
{
	double pdf;
	double z;
	double grad;
	pdf  = dlogis(x,mu,sd);
	z    = (x-mu)/sd;
	grad = pdf*(1-std::pow(std::exp(1),z))/(1+std::pow(std::exp(1),z));
    return grad;
}