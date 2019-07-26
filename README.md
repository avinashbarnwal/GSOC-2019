# GSOC-2019

# Learning Internals of Xgboost

Following is the sequence of files:-

- xgb_UpdateOneIter.cc - It has GetGradient and DoBoost.
- xgb_GetGradient.cc   - It has loss functions first order and second order methods.
- xgboost_lr_obj.cc    - It has objective functions methods.
- xgb_DoBoost.cc       - It does boosting.
- xgb_gbtree_update.cc - It updates boosting cycle and updation of trees.
- xgb_grow_colmaker.cc - Each tree is updated by the builder depth by depth.
- xgb_updater_prune.cc - It prunes tree leaves recursively.
- xgboost_loss_chg.cc  - It calculates the change in loss.
- xgb_CalcGain.cc      - It calculates the gain for each tree node.
- xgb_CalcWeight.cc    - It calculates weight of each tree node.
- xgb_PredValue.cc     - It calculates the prediction for new data point.

**UpdateOneIter** -> **Gradient and Hessian** -> **Loss** -> **DoBoost** -> **Update Boosting Cycle** -> **Update tree depth by depth** -> **Tree Pruning** -> **Calculate Gain** -> **Calculate weight of node**  

- Link - https://towardsdatascience.com/boosting-algorithm-xgboost-4d9ec0207d



# Makefile -
- https://opensource.com/article/18/8/what-how-makefile

# How to wrap the C++ code in Python  
- Using shared library and python library ctypes.  


# Changes in the files to include AFT in Xgboost  
- **src/c_api/c_api.cc  **
Code Below  
  vec = &info.weights_.HostVector();  
  } else if (!std::strcmp(field, "base_margin")) {  
    vec = &info.base_margin_.HostVector();  
  } else if (!std::strcmp(field, "label_lower_bound")) {  
    vec = &info.labels_lower_bound_.HostVector();  
  } else if (!std::strcmp(field, "label_upper_bound")) {  
    vec = &info.labels_upper_bound_.HostVector();  
  } else {  
    LOG(FATAL) << "Unknown float field name " << field;  
  }  
- src/data/data.cc  
 Code Below  
    labels.resize(num);  
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,  
                       std::copy(cast_dptr, cast_dptr + num, labels.begin()));  
  } else if (!std::strcmp(key, "label_lower_bound")) {  
    auto& labels = labels_lower_bound_.HostVector();  
    labels.resize(num);  
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,  
                       std::copy(cast_dptr, cast_dptr + num, labels.begin()));  
  } else if (!std::strcmp(key, "label_upper_bound")) {  
    auto& labels = labels_upper_bound_.HostVector();  
    labels.resize(num);  
    DISPATCH_CONST_PTR(dtype, dptr, cast_dptr,  
                       std::copy(cast_dptr, cast_dptr + num, labels.begin()));  
  } else if (!std::strcmp(key, "weight")) {  
    auto& weights = weights_.HostVector();  
    weights.resize(num);  
 - include/xgboost/data.h  
  uint64_t num_nonzero_{0};  
  /*! \brief label of each instance */  
  HostDeviceVector<bst_float> labels_;  
  /*! \brief lower bound label of each instance; used for survival analysis,  
   *         where labels are left-, right- or interval-censored. */  
  HostDeviceVector<bst_float> labels_lower_bound_;  
  /*! \brief upper bound label of each instance; used for survival analysis,  
   *         where labels are left-, right- or interval-censored. */  
  HostDeviceVector<bst_float> labels_upper_bound_;  
  /*!
   * \brief specified root index of each instance,  
   *  can be used for multi task setting  
@@ -68,9 +74,11 @@ class MetaInfo {  
   */  
  HostDeviceVector<bst_float> base_margin_;  
  /*! \brief version flag, used to check version of this info */  
  static const int kVersion = 2;  
  /*! \brief version that introduced qid field */  
  static const int kVersion = 3;  
  /*! \brief version that introduced field qids_ */  
  static const int kVersionQidAdded = 2;  
  /*! \brief version that introduced fields labels_lower_bound_, labels_upper_bound_ */  
  static const int kVersionBounedLabelAdded = 3;  
  /*! \brief default constructor */  
  MetaInfo()  = default;  
  /*!  
  
 - src/data/data.cc
 @@ -42,6 +42,8 @@ void MetaInfo::SaveBinary(dmlc::Stream *fo) const {  
  fo->Write(labels_.HostVector());  
  fo->Write(group_ptr_);  
  fo->Write(qids_);  
  fo->Write(labels_lower_bound_.HostVector());  
  fo->Write(labels_upper_bound_.HostVector());  
  fo->Write(weights_.HostVector());  
  fo->Write(root_index_);  
  fo->Write(base_margin_.HostVector());  
@@ -59,9 +61,17 @@ void MetaInfo::LoadBinary(dmlc::Stream *fi) {  
  CHECK(fi->Read(&group_ptr_)) << "MetaInfo: invalid format";  
  if (version >= kVersionQidAdded) {  
    CHECK(fi->Read(&qids_)) << "MetaInfo: invalid format";  
  } else {  // old format doesn't contain qid field  
  } else {  // old format doesn't contain field qids_  
    qids_.clear();  
  }  
  if (version >= kVersionBounedLabelAdded) {  
    CHECK(fi->Read(&labels_lower_bound_.HostVector())) << "MetaInfo: invalid format";  
    CHECK(fi->Read(&labels_upper_bound_.HostVector())) << "MetaInfo: invalid format";  
  } else {  // old format doesn't contain fields labels_lower_bound_, labels_upper_bound_  
    qids_.clear();  
    labels_lower_bound_.HostVector().clear();  
    labels_upper_bound_.HostVector().clear();  
  }  
  CHECK(fi->Read(&weights_.HostVector())) << "MetaInfo: invalid format";  
  CHECK(fi->Read(&root_index_)) << "MetaInfo: invalid format";  
  CHECK(fi->Read(&base_margin_.HostVector())) << "MetaInfo: invalid format";  
