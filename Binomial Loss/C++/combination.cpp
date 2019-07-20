#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"


extern "C" double combination(double N, double R) 
{ 
return (factorial(N))/((factorial(N-R))*factorial(R)); 
} 
