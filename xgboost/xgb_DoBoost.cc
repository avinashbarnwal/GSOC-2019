void DoBoost(DMatrix* p_fmat, std::vector<bst_gpair>* in_gpair,
             ObjFunction* obj) override {
  const std::vector<bst_gpair>& gpair = *in_gpair;
  std::vector<std::vector<std::unique_ptr<RegTree> > > new_trees;
  // for binary classification task
  if (mparam.num_output_group == 1) {
    std::vector<std::unique_ptr<RegTree> > ret;
    BoostNewTrees(gpair, p_fmat, 0, &ret);
    new_trees.push_back(std::move(ret));
  } else {
  // others
  }
}

// do group specific group
inline void BoostNewTrees(const std::vector<bst_gpair> &gpair, DMatrix *p_fmat, 
                          int bst_group, std::vector<std::unique_ptr<RegTree> >* ret) {
  this->InitUpdater();
  std::vector<RegTree*> new_trees;
  // create the trees
  // for boosting, num_parallel_tree equals to 1
  for (int i = 0; i < tparam.num_parallel_tree; ++i) {
    if (tparam.process_type == kDefault) {
      // create new tree
      std::unique_ptr<RegTree> ptr(new RegTree());
      ptr->param.InitAllowUnknown(this->cfg);
      ptr->InitModel();
      new_trees.push_back(ptr.get());
      ret->push_back(std::move(ptr));
    } else if (tparam.process_type == kUpdate) {
      // update the existing tree
    }
  }
  // update the trees
  for (auto& up : updaters) {
    up->Update(gpair, p_fmat, new_trees);
  }
}