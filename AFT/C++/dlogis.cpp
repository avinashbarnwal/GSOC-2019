#include <iostream>
#include <math.h>
#define PI 3.14159
using namespace std;
double dlogis(double x, double mu, double sd);

int main()
{
   double x    = 0;
   double mu   = 0;
   double sd   = 1;
   double res;
   res = dlogis(x,mu,sd);
   return 0;

}

double dlogis(double x, double mu , double sd)
{
	double pdf;
  pdf = exp((x-mu)/sd)/(sd*pow((1+exp((x-mu)/sd)),2));
  return pdf;

}