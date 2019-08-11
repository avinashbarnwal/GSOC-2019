#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"


extern "C" double pextreme(double x, double mu , double sd)
{
	double cdf;
	double z;
	double w;
	z = (x-mu)/sd;
  	w = std::exp(z);
    cdf =  1-std::exp(-w);
    return cdf;
}
