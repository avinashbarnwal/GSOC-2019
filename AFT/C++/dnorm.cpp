#include <iostream>
#include <cmath>
#define PI 3.14159

extern "C" double dnorm(double x, double mu , double sd)
{
	double pdf;
    pdf = (std::exp(-std::pow((x-mu)/(std::sqrt(2)*sd),2)))/std::sqrt(2*PI*std::pow(sd,2));
    return pdf;

}
