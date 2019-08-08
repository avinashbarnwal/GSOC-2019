#include <iostream>
#include <cmath>
#define PI 3.14159
#include "aft.h"

extern "C" double dlogis(double x, double mu , double sd)
{
  double pdf;
  pdf = std::exp((x-mu)/sd)/(sd*std::pow((1+std::exp((x-mu)/sd)),2));
  return pdf;

}
