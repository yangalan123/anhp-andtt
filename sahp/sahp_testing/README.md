# Testing package for SAHP

[//]: # (work-in-progress, please check back later)

This sub-package is used to do testing for SAHP. We implemented thinning algorithm (see `train_functions/train_sahp.thinning`) using SAHP codes for a principled and fair comparison with our models as explained in our paper. 
The code structure in this sub-package is very similar to `sahp_training`. So you can follow the same instruction to run testing here. The training is disabled by setting `epochs = 0` in `main_func.py` by default.
