#include <iostream>
#include <math.h>
#define PI 3.14159
using namespace std;

double pnorm(double x, double mu , double sd)
{
	double cdf;
    cdf = 0.5*(1+erf((x-mu)/(sd*sqrt(2))));
    return cdf;
}
