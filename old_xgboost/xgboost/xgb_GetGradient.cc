void GetGradient(const std::vector<bst_float> &preds,
                 const MetaInfo &info,
                 int iter, std::vector<bst_gpair> *out_gpair) override {
  // start calculating gradient
  const omp_ulong ndata = static_cast<omp_ulong>(preds.size());
  for (omp_ulong i = 0; i < ndata; ++i) {
    bst_float p = Loss::PredTransform(preds[i]);
    bst_float w = info.GetWeight(i);
    if (info.labels[i] == 1.0f) w *= param_.scale_pos_weight;
    if (!Loss::CheckLabel(info.labels[i])) label_correct = false;
    out_gpair->at(i) = bst_gpair(Loss::FirstOrderGradient(p, info.labels[i]) * w,
                                 Loss::SecondOrderGradient(p, info.labels[i]) * w);
  }
}