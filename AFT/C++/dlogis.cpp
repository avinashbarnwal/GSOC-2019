#include <iostream>
#include <math.h>
#define PI 3.14159
using namespace std;

double dlogis(double x, double mu , double sd)
{
	double pdf;
  pdf = exp((x-mu)/sd)/(sd*pow((1+exp((x-mu)/sd)),2));
  return pdf;

}
