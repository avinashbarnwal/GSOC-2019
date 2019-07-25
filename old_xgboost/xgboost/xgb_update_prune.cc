/*! \brief do pruning of a tree */
inline void DoPrune(RegTree &tree) { // NOLINT(*)
  int npruned = 0;
  for (int nid = 0; nid < tree.param.num_nodes; ++nid) {
    if (tree[nid].is_leaf()) {
      npruned = this->TryPruneLeaf(tree, nid, tree.GetDepth(nid), npruned);
    }
  }
}

// try to prune off current leaf
inline int TryPruneLeaf(RegTree &tree, int nid, int depth, int npruned) { // NOLINT(*)
  if (s.leaf_child_cnt >= 2 && param.need_prune(s.loss_chg, depth - 1)) {
    // need to be pruned
    tree.ChangeToLeaf(pid, param.learning_rate * s.base_weight);
    // tail recursion
    return this->TryPruneLeaf(tree, pid, depth - 1, npruned + 2);
  } else {
    return npruned;
  }
}

/*! \brief given the loss change, whether we need to invoke pruning */
inline bool need_prune(double loss_chg, int depth) const {
  return loss_chg < this->min_split_loss;
}
