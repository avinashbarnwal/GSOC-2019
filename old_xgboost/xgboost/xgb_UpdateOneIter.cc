void UpdateOneIter(int iter, DMatrix* train) override {
  this->LazyInitDMatrix(train);
  this->PredictRaw(train, &preds_);
  obj_->GetGradient(preds_, train->info(), iter, &gpair_);
  gbm_->DoBoost(train, &gpair_, obj_.get());
}