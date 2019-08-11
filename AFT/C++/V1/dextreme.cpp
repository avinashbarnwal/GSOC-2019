#include <iostream>
#include <cmath>
#define PI 3.14159
#include "aft.h"

extern "C" double dextreme(double x, double mu , double sd)
{
  double pdf;
  double z;
  double w;
  z = (x-mu)/sd;
  w = std::exp(z);
  pdf =  w*std::exp(-w);
  return pdf;

}
