#include <iostream>
#include <cmath>
#include <string>
#define PI 3.14159
#include "aft.h"

extern "C" double hessian_interval(double y_lower,double y_higher,double y_pred,double sigma,char* dist)
{
  double z_u;
  double z_l;
  double pdf_u;
  double pdf_l;
  double cdf_u;
  double cdf_l;
  double grad_u;
  double grad_l;
  double hess;
  char* given_dist1 =  "normal";
  char* given_dist2 =  "logistic";
  char* given_dist3 =  "extreme";

  z_u   = (std::log(y_higher) - y_pred)/sigma;
  z_l   = (std::log(y_lower) - y_pred)/sigma;
  if(strcmp(dist, given_dist1) == 0){
    pdf_u   = dnorm(z_u,0,1);
    pdf_l   = dnorm(z_l,0,1);
  	cdf_u   = pnorm(z_u,0,1);
    cdf_l   = pnorm(z_l,0,1);
    grad_u  = grad_norm(z_u,0,1);
    grad_l  = grad_norm(z_l,0,1);
  }
  else if(strcmp(dist, given_dist2) == 0) {
    pdf_u = dlogis(z_u,0,1);
    pdf_l = dlogis(z_l,0,1);
  	cdf_u = plogis(z_u,0,1);
    cdf_l = plogis(z_l,0,1);
    grad_u  = grad_logis(z_u,0,1);
    grad_l  = grad_logis(z_l,0,1);
  }
  else if(strcmp(dist, given_dist3) == 0) {
    pdf_u = dextreme(z_u,0,1);
    pdf_l = dextreme(z_l,0,1);
    cdf_u = pextreme(z_u,0,1);
    cdf_l = pextreme(z_l,0,1);
    grad_u  = grad_extreme(z_u,0,1);
    grad_l  = grad_extreme(z_l,0,1);
  }
  hess = -((cdf_u-cdf_l)*(grad_u-grad_l)-std::pow((pdf_u-pdf_l),2))/(std::pow(sigma,2)*std::pow((cdf_u-cdf_l),2)); 
  return hess;
}
