loss_chg = static_cast<bst_float>(constraints_[nid].CalcSplitGain(param, fid, e.stats, c) - snode[nid].root_gain);
snode[nid].root_gain = static_cast<float>(constraints_[nid].CalcGain(param, snode[nid].stats));

inline double CalcSplitGain(const TrainParam &param, bst_uint split_index,
                            GradStats left, GradStats right) const {
  return left.CalcGain(param) + right.CalcGain(param);
}