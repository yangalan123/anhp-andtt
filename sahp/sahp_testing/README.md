# Testing package for SAHP

[//]: # (work-in-progress, please check back later)

This sub-package is used to do testing for SAHP. We implemented thinning algorithm (see `train_functions/train_sahp.thinning`) using SAHP codes for a principled and fair comparison with our models as explained in our paper. 
The code structure in this sub-package is very similar to `sahp_training`. So you can follow the same instruction to run testing here. The training is disabled by setting `epochs = 0` in `main_func.py` by default.

You should use the same `log_dir` as you use when training the SAHP model to run the evaluation codes as the evaluation process is called in `train_functions.train_sahp.train_eval_sahp`. When `epochs==0`, then no training will be executed, just doing evaluation using thinning algorithm. 

You can also check out `gen_data.py` for generating data from a SAHP model to do synthetic experiments as shown in our ICLR'22 paper. 
