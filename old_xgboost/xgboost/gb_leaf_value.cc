inline void Refresh(const TStats *gstats,
                    int nid, RegTree *p_tree) {
  if (tree[nid].is_leaf()) {
    if (param.refresh_leaf) {
      tree[nid].set_leaf(tree.stat(nid).base_weight * param.learning_rate);
    }
  }
}