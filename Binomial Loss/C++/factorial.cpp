#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"

extern "C" double factorial(double val) { 
	double res = 1; 
	for(int i = 1; i <= val; i++) { 
	res *= i; 
	} 
return res;
} 