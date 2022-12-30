# Testing Package for THP

[//]: # (work-in-progress, please check back later)

This package implements the testing and evaluation for GaTech THP model. 
We follow our own code structure to re-implement GaTech codes to make sure we can use thinning algorithm to do principled and fair comparison between models as explained in our paper.
After you have used the `thp_training` sub-packages to finish the training, you should be able to run testing using this sub-package. 
The running instruction is pretty similar to ANHP/ANDTT as they share similar code structure.

### Test Models
Go to the `run` directory.

To test the trained model, use the command line below for detailed guide:
```
python test.py --help
```

Example command line for testing:

```
python test.py -d YOUR_DOMAIN -fn FOLDER_NAME -s test -sd 12345 -pred
```

To evaluate the model predictions, use the command line below for detailed guide:
```
python eval.py --help
```

Example command line for testing:

```
python eval.py -d YOUR_DOMAIN -fn FOLDER_NAME -s test
```
