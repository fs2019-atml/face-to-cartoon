# Conditional cartoon faces incorporating a landmark loss in CycleGAN

## Report
The `Report_FaceToCartoon.ipynb ` covers the details of the project.

## Demo
You find an interactive demo under `code/Demo.ipynb`.

## Code
More details to the code are in `code/README.md`.

## How to train
1) Clone this repo using `git clone https://github.com/fs2019-atml/face-to-cartoon.git`
2) Get the dataset from google drive `https://drive.google.com/open?id=12vU_Dkn13KqsVy5LOYlpoTJGWmzPqPQh`
3) Untar the archive to `./code/datasets/` to have all the images under `./code/datasets/faces/{cartoon/*, real/*}` (e.g. `tar xf faces.tar.gz` inside `./code/datasets`)
4) Install dependencies (See below)
5) a) Invoke `python train.py --gpu_ids=0` to train on cuda. b) Invoke `python train.py` to wait forever.

### Dependencies
We suggest an anaconda environment with the following:
```
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install pytorch torchvision -c pytorch # add cuda90 if CUDA 9 (or magma-cuda90)
conda install visdom dominate -c conda-forge # install visdom and dominate
```

If you run in troubles with Cuda try to downgrade pytorch to version 0.41.

