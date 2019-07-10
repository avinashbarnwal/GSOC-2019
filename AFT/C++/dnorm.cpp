#include <iostream>
#include <math.h>
#define PI 3.14159
using namespace std;

double dnorm(double x, double mu , double sd)
{
	double pdf;
    pdf = (exp(-pow((x-mu)/(sqrt(2)*sd),2)))/sqrt(2*PI*pow(sd,2));
    return pdf;

}
