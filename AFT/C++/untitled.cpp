#include <iostream>
#include <cmath>
#define PI 3.14159

double dnorm(double x, double mu , double sd)
{
	double pdf;
    pdf = (std::exp(-std::pow((x-mu)/(std::sqrt(2)*sd),2)))/std::sqrt(2*PI*std::pow(sd,2));
    return pdf;

}

double _grad_norm(z)
{
	double pdf;
	pdf  = dnorm(z);
	grad = -z*pdf;
    return grad;
}


int main(){

	double z = 0;
	double grad;
	grad = _grad_norm(z);
	std::cout << grad << endl;

}