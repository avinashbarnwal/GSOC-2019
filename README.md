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

# LightGBM vs Xgboost
- It introduces two techniques to improve performance. 
- Gradient-based One-Side Sampling which inspects the most informative samples while skipping the less informative samples.
- Exclusive Feature Bundling which takes advantage of sparse datasets by grouping features in a near lossless way.
