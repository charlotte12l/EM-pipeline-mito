# EM Mitochondria Pipeline

This pipeline is derived from [zudi-lin/pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics), I did some modification and new implementation:

### Pre-process and Post-process
- get_dataset.py: Get .h5 volume dataset from .png proofreading files and EM images.
- get_seg.py: Use CC and watershed to get mitochondria segmentation map from a distance transform map, and calculate the precision and recall for the segmentation. derived from [aarushgupta/NeuroG/post_process.py](https://github.com/aarushgupta/NeuroG)
- get_vast_deploy_HUMAN.py: Large scale version of get_seg.py. After getting the heatmap results(20 slices each), merge them to 200 slices each and do post-processing.
- trace.py: After using get_vast_deploy_HUMAN.py to get the segmentation. The labels will not be consistent through the 1200 slices. This script help to trace the mitochondria and relabel them.
- Upsample.py: Upsample the large volume for better visualization in neuroglancer. For neuroglancer visualization, please refer to [aarushgupta/NeuroG/post_process.py](https://github.com/aarushgupta/NeuroG)
- T_util.py: Some useful functions
- torch_connectomics/run/genertae_slurm.py: Automatically generate slurm scripts for running inference on RC cluster 
- torch_connectomics/run/deploy.py: Use trained model on large scale dataset to get heatmaps.

### Online Hard Negative Mining:
- scripts/train.py: Training script for OHEM, please set args.aux and criterion for OHEM.
- torch_connectomics/run/train.py: Train function for OHEM
- torch_connectomics/model/model_zoo/unetv3.py: new unetv3 version, with auxiliary outputs
- torch_connectomics/model/loss/loss.py: implementation of different OHEM loss
- torch_connectomics/utils/vis/visualize.py: also visualize the auxiliary outputs and gts 

### Nearest Neighbor Classification
- scripts/train_NN.py: Training script for NNC
- torch_connectomics/run/train_NN.py:Train function for NNC
- torch_connectomics/run/test_NN.py:Test function for NNC
- torch_connectomics/model/model_zoo/unetv3_ebd2.py:Model for 3D UNet+NNC, get embedding from the last but one layer

### Discriminative Loss 
- scripts/train_dsc.py:Training script for Discriminative Loss
- torch_connectomics/run/train_dsc.py:Train function for Discriminative Loss
- torch_connectomics/run/test_dsc.py:Test function for Discriminative Loss
- torch_connectomics/model/loss/loss.py: implementation of Discriminative Loss

### Embedding
Implentation for 
- scripts/train_2D.py:Training script for 2D UNet+Embedding
- torch_connectomics/run/train_2d.py:Train function for 2D UNet+Embedding
- torch_connectomics/model/model_zoo/unet2d_ebd.py: Model for 2D UNet+Embedding
- torch_connectomics/model/model_zoo/unetv3_ebd.py: Model for 3D UNet+Embedding

### Training & Testing in one file
The past code requires using scripts/test.py to evaluate the trained model, but here I implement automatically evaluate the model after every 1000 iterations
- scripts/train_val.py: Scripts for training & testing
- torch_connectomics/run/train_val.py: Function for training & testing
- torch_connectomics/utils/net/arguments_val.py: Arguments for training & testing
- torch_connectomics/utils/net/dataloader_val.py: Dataloader for training & testing

                                            Please install the package as follows:
-------------------------------------------------------------------------------------------------------------------------
# PyTorch Connectomics

## Introduction

The field of connectomics aims to reconstruct the wiring diagram of the brain by mapping the neural connections at the level of individual synapses. Recent advances in electronic microscopy (EM) have enabled the collection of a large number of image stacks at nanometer resolution, but the annotation requires expertise and is super time-consuming. Here we provide a deep learning framework powered by [PyTorch](https://pytorch.org/) for automatic and semi-automatic data annotation in connectomics. This repository is actively under development by Visual Computing Group ([VCG](https://vcg.seas.harvard.edu)) at Harvard University.

## Key Features

- Multitask Learning
- Active Learning
- CPU and GPU Parallelism

If you want new features that are relatively easy to implement (e.g. loss functions, models), please open a feature requirement discussion in issues or implement by yourself and submit a pull request. For other features that requires substantial amount of design and coding, please contact the [author](https://github.com/zudi-lin) directly. 

## Environment

The code is developed and tested under the following configurations.
- Hardware: 1-8 Nvidia GPUs (with at least 12G GPU memories) (change ```[-g NUM_GPU]``` accordingly)
- Software: CentOS Linux 7.4 (Core), ***CUDA>=9.0, Python>=3.6, PyTorch>=1.0.0***

## Installation

Create a new conda environment:
```
conda create -n py3_torch python=3.6
source activate py3_torch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

Download and install the package:
```
git clone git@github.com:zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install -r requirements.txt
pip install --editable .
```
For more information and frequently asked questions about installation, please check the [installation guide](https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/installation.html). If you meet compilation errors, please check [TROUBLESHOOTING.md](https://github.com/zudi-lin/pytorch_connectomics/blob/master/TROUBLESHOOTING.md).

## Visulazation

### Training
* Visualize the training loss and validation images using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).

### Test
* Visualize the affinity graph and segmentation using Neuroglancer.

## Notes

### Data Augmentation
We provide a data augmentation interface several different kinds of commonly used augmentation method for EM images. The interface is pure-python, and operate on and output only numpy arrays, so it can be easily incorporated into any kinds of python-based deep learning frameworks (e.g. TensorFlow). For more details about the design of the data augmentation module, please check the [documentation](https://zudi-lin.github.io/pytorch_connectomics/build/html/modules/augmentation.html).

### Model Zoo
We provide several encoder-decoder architectures, which can be found [here](https://github.com/zudi-lin/pytorch_connectomics/tree/master/torch_connectomics/model/model_zoo). Those models can be applied to any kinds of semantic segmentation tasks of 3D image stacks. We also provide benchmark results on SNEMI3D neuron segmentation challenges [here](https://github.com/zudi-lin/pytorch_connectomics/tree/master/benchmark) with detailed training specifications for users to reproduce.

### Syncronized Batch Normalization on PyTorch
Previous works have suggested that a reasonable large batch size can improve the performance of detection and segmentation models. Here we use a syncronized batch normalization module that computes the mean and standard-deviation across all devices during training. Please refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) for details. The implementation is pure-python, and uses unbiased variance to update the moving average, and use `sqrt(max(var, eps))` instead of `sqrt(var + eps)`.

## Acknowledgement
This project is built upon numerous previous projects. Especially, we'd like to thank the contributors of the following github repositories:
- [pyGreenTea](https://github.com/naibaf7/PyGreentea): Janelia FlyEM team 
- [DataProvider](https://github.com/torms3/DataProvider): Princeton SeungLab
- [EM-affinity](https://github.com/donglaiw/EM-affinity): Harvard Visual Computing Group

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE) file for details.

## Contact
[Zudi Lin](https://github.com/zudi-lin)
