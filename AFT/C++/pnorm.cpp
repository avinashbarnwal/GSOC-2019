#include <iostream>
#include <cmath>
#define PI 3.14159

double pnorm(double x, double mu , double sd)
{
	double cdf;
    cdf = 0.5*(1+std::erf((x-mu)/(sd*std::sqrt(2))));
    return cdf;
}
