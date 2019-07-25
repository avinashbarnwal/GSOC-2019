// predict the leaf scores without dropped trees
inline bst_float PredValue(const RowBatch::Inst &inst,
                           int bst_group,
                           unsigned root_index,
                           RegTree::FVec *p_feats,
                           unsigned tree_begin,
                           unsigned tree_end) {
  bst_float psum = 0.0f;
  p_feats->Fill(inst);
  for (size_t i = tree_begin; i < tree_end; ++i) {
    if (tree_info[i] == bst_group) {
      bool drop = (std::binary_search(idx_drop.begin(), idx_drop.end(), i));
      if (!drop) {
        int tid = trees[i]->GetLeafIndex(*p_feats, root_index);
        psum += weight_drop[i] * (*trees[i])[tid].leaf_value();
      }
    }
  }
  p_feats->Drop(inst);
  return psum;
