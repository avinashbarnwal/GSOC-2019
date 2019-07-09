// calculate weight given the statistics
template <typename TrainingParams, typename T>
XGB_DEVICE inline T CalcWeight(const TrainingParams &p, T sum_grad,
                               T sum_hess) {
  if (sum_hess < p.min_child_weight)
    return 0.0;
  T dw;
  if (p.reg_alpha == 0.0f) {
    dw = -sum_grad / (sum_hess + p.reg_lambda);
  } else {
    dw = -ThresholdL1(sum_grad, p.reg_alpha) / (sum_hess + p.reg_lambda);
  }
  if (p.max_delta_step != 0.0f) {
    if (dw > p.max_delta_step)
      dw = p.max_delta_step;
    if (dw < -p.max_delta_step)
      dw = -p.max_delta_step;
  }
  return dw;
}