# Introduction
This is the bug-fixed version for [Transformer Hawkes Process (THP)](https://arxiv.org/abs/2002.09291).

We have contacted the first author of the THP paper, Simiao Zuo for the confirmation of the bug and our fixes. 

## Package Description
1. For training, please go into `thp_training` sub-package and follow the instructions there. You will obtain the model `model.pt` that can be used for testing purposes. 
Please note that the RMSE and Accuracy stuffs in the training log is buggy.
For evaluation purposes (computing loglik, RMSE, Accuracy), please use `thp_testing` sub-package, which will be introduced below.

2. For testing and synthetic data drawing, as the THP authors do not provide thinning algorithm implementation, for a fair comparison, we create a separate sub-package `thp_testing` to do that. 
This sub-package is still under construction and will be released soon.