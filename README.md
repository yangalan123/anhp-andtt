# Maintainance Status: Archival Mode (2024)
If you are seeking more up-to-date efficient implementations of ANHP, THP, and SAHP, I recommend checking out [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess). They also provide better community support for your TPP usage. This repository is currently under archival (Read-Only) mode. 

# Introduction

Codebase for the paper [**Transformer Embeddings of Irregularly Spaced Events and Their Participants**](https://arxiv.org/abs/2201.00044).

Author: Chenghao Yang (yangalan1996@gmail.com)

This codebase contains several packages:
1. `anhp`: Attentive-Neural Hawkes Process (A-NHP)
2. `andtt`: Attentive-Neural Datalog Through Time (A-NDTT). 
3. `thp`: Our corrected version of [Transformer Hawkes Process (THP)](https://arxiv.org/abs/2002.09291).
4. `sahp`: Our corrected version of [Self-Attentive Hawkes Process (SAHP)](https://arxiv.org/abs/1907.07561).

For `thp` and `sahp`, our code includes certain corrections that have been discussed with and agreed by the authors of those papers.

## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{yang-2021-transformer,
  author =      {Chenghao Yang and Hongyuan Mei and Jason Eisner},
  title =       {Transformer Embeddings of Irregularly Spaced Events and Their Participants},
  booktitle =     {International Conference on Learning Representations},
  year =        {2022}
}
```


## Instructions
Here are the instructions to use the code base.

### Dependencies and Installation
This code is written in Python 3, and I recommend you to install:
* [Anaconda](https://www.continuum.io/) that provides almost all the Python-related dependencies;

This project relies on Datalog Utilities in [NDTT project](https://github.com/hongyuanmei/neural-datalog-through-time), please first install it.
(**please remove the `torch` version (`1.1.0`) in `setup.py` of NDTT project, because that is not the requirement of this project and we only use non-pytorch part of NDTT. We recommend using `torch>=1.7` for this project.**).

Then run the command line below to install the package (add `-e` option if you need an editable installation):
```
pip install .
```

### Dataset Preparation
Download datasets and programs from [here](https://drive.google.com/drive/folders/17vtQdx3d1wR-SADSMamt4E2mqHfEOu9q).

Organize your domain datasets as follows:
```
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA
```

### (A-NDTT-only) Build Dynamic Databases
Go to the `andtt/run` directory. 

To build the dynamic databases for your data, try the command line below for detailed guide: 
```
python build.py --help
```

The generated dynamic model architectures (represented by database facts) are stored in this directory: 
```
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA/tdbcache
```


### Train Models
To train the model specified by your Datalog probram, try the command line below for detailed guide:
```
python train.py --help
```

The training log and model parameters are stored in this directory: 
```
# A-NHP
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA/ContKVLogs
# A-NDTT
domains/YOUR_DOMAIN/YOUR_PROGRAMS_AND_DATA/Logs
```

Example command line for training:
```
# A-NHP
python train.py -d YOUR_DOMAIN -ps ../../ -bs BATCH_SIZE -me 50 -lr 1e-4 -d_model 32 -teDim 10 -sd 1111 -layer 1
# A-NDTT
python train.py -d YOUR_DOMAIN -db YOUR_PROGRAM -ps ../../ -bs BATCH_SIZE -me 50 -lr 1e-4 -d_model 32 -teDim 10 -sd 1111 -layer 1
```

### Test Models
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
1. The transformer component implementation used in this repo is based on widely-recognized [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html
). 
1. The code structure is inspired by Prof. Hongyuan Mei's [Neural Datalog Through Time](https://github.com/HMEIatJHU/neural-datalog-through-time.git)



