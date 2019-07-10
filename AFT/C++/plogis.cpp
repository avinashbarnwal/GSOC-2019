#include <iostream>
#include <math.h>
#define PI 3.14159
using namespace std;

double plogis(double x, double mu , double sd)
{
	double cdf;
    cdf = exp((x-mu)/sd)/(1+exp((x-mu)/sd));
    return cdf;
}
