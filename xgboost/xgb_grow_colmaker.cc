// inside ColMaker
void Update(const std::vector<bst_gpair> &gpair, DMatrix* dmat, 
            const std::vector<RegTree*> &trees) override {
  // build tree
  for (size_t i = 0; i < trees.size(); ++i) {
    Builder builder(param);
    builder.Update(gpair, dmat, trees[i]);
  }
}

// Update method of builder
virtual void Update(const std::vector<bst_gpair>& gpair, DMatrix* p_fmat, RegTree* p_tree) {
  this->InitData(gpair, *p_fmat, *p_tree);
  this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
  for (int depth = 0; depth < param.max_depth; ++depth) {
    this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
    this->ResetPosition(qexpand_, p_fmat, *p_tree);
    this->UpdateQueueExpand(*p_tree, &qexpand_);
    this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
    // if nothing left to be expand, break
    if (qexpand_.size() == 0) break;
  }
}