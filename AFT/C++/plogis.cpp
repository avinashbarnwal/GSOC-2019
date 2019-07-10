#include <iostream>
#include <cmath>
#define PI 3.14159

extern "C" double plogis(double x, double mu , double sd)
{
	double cdf;
    cdf = std::exp((x-mu)/sd)/(1+std::exp((x-mu)/sd));
    return cdf;
}
