# Introduction
This is the corrected version for [Self-Attentive Hawkes Process (SAHP)](https://arxiv.org/abs/1907.07561).

We have discussed with the first author of the SAHP paper, Qiang Zhang for the confirmation and agreement of the corrections. 

## Package Description
1. For training, please go into `sahp_training` sub-package and follow the instructions there. You will obtain the model `model.pt` that can be used for testing purposes.
   Please note that the RMSE and Accuracy stuffs in the training log is not reliable.
   For evaluation purposes (computing loglik, RMSE, Accuracy), please use `sahp_testing` sub-package, which will be introduced below.

2. For testing and synthetic data drawing, as the SAHP authors do not provide thinning algorithm implementation, for a fair comparison, we create a separate sub-package `sahp_testing` to do that.
