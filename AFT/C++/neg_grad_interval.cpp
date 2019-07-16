#include <iostream>
#include <cmath>
#define  PI 3.14159
#include "aft.h"

extern "C" double neg_grad_interval(double y_lower,double y_higher,double y_pred,double sigma,char* dist)
{
	double pdf_l;
  double pdf_u;
	double z_u;
  double z_l;
	double cdf_u;
  double cdf_l;
	double neg_grad;
  char* given_dist =  "normal";
	z_u    = (std::log(y_higher)-y_pred)/sigma;
  z_l    = (std::log(y_lower)-y_pred)/sigma;
  if(strcmp(dist, given_dist) == 0){
  		pdf_u  = dnorm(z_u,0,1);
      pdf_l  = dnorm(z_l,0,1);
      cdf_u  = pnorm(z_u,0,1);
  		cdf_l  = pnorm(z_l,0,1);
  }
  else{
  		pdf_u  = dlogis(z_u,0,1);
      pdf_l  = dlogis(z_l,0,1);
      cdf_u  = plogis(z_u,0,1); 
  		cdf_l  = plogis(z_l,0,1); 
  }
  neg_grad  = -(pdf_u-pdf_l)/(sigma*std::max(0.00005,cdf_u-cdf_l));

  return neg_grad;
}
