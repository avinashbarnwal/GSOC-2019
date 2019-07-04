#include <iostream>
#include <math.h>
#define PI 3.14159
using namespace std;
double plogis(double x, double mu, double sd);
int main()
{
   double x    = 0;
   double mu   = 0;
   double sd   = 1;
   double res;
   res = plogis(x,mu,sd);
   return 0;

}

double plogis(double x, double mu , double sd)
{
	double cdf;
    cdf = exp((x-mu)/sd)/(1+exp((x-mu)/sd));
    return cdf;
}