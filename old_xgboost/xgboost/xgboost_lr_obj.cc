XGBOOST_REGISTER_OBJECTIVE(LogisticClassification, "binary:logistic")
.describe("Logistic regression for binary classification task.")
.set_body([]() { return new RegLossObj<LogisticClassification>(); });

// logistic loss for probability regression task
struct LogisticRegression {
  static bst_float PredTransform(bst_float x) { return common::Sigmoid(x); }
  
  // the objective function should provide the first order gradient and the second order gradient
  static bst_float FirstOrderGradient(bst_float predt, bst_float label) { return predt - label; }
  static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    const float eps = 1e-16f;
    return std::max(predt * (1.0f - predt), eps);
  }
};

// logistic loss for binary classification task.
struct LogisticClassification : public LogisticRegression {
  static const char* DefaultEvalMetric() { return "error"; }
};