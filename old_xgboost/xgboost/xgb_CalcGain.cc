template <typename TrainingParams, typename T>
XGB_DEVICE inline T CalcGain(const TrainingParams &p, T sum_grad, T sum_hess) {
  if (sum_hess < p.min_child_weight)
    return 0.0;
  if (p.max_delta_step == 0.0f) {
    if (p.reg_alpha == 0.0f) {
      return Sqr(sum_grad) / (sum_hess + p.reg_lambda);
    } else {
      return Sqr(ThresholdL1(sum_grad, p.reg_alpha)) /
             (sum_hess + p.reg_lambda);
    }
  } else {
    T w = CalcWeight(p, sum_grad, sum_hess);
    T ret = sum_grad * w + 0.5 * (sum_hess + p.reg_lambda) * Sqr(w);
    if (p.reg_alpha == 0.0f) {
      return -2.0 * ret;
    } else {
      return -2.0 * (ret + p.reg_alpha * std::abs(w));
    }
  }
}