// initialize updater before using them
inline void InitUpdater() {
  if (updaters.size() != 0) return;
  // updater_seq is the string defining the sequence of tree updaters
  // default is set as grow_colmaker,prune
  std::string tval = tparam.updater_seq;
  std::vector<std::string> ups = common::Split(tval, ',');
  for (const std::string& pstr : ups) {
    std::unique_ptr<TreeUpdater> up(TreeUpdater::Create(pstr.c_str()));
    up->Init(this->cfg);
    updaters.push_back(std::move(up));
  }
}