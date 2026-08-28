# NeSF

NeSF fork of jax3d. This codebase contains the original implementation of neural
semantic fields, a method for 3D semantic segmentation from posed 2D inputs and
supervision. Learn more at https://nesf3d.github.io/.

# (Optional) Environment

Prior to installation, it can be a good practice to set up virtual environments.
For example, consider installing Miniconda e.g.:

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Using conda, create and activate a virtual environment, which will allow safe
installation of dependencies (based on Python 3.10.8):

```shell
conda create -n nesf
conda activate nesf
conda install python
```

# Installation

NeSF requires additional libraries other than those required for `jax3d`. The
following installs those libraries and support for single- and multi-GPU
training.

To begin, clone the repository.

```shell
git clone https://github.com/google-research/jax3d.git
```

From within the jax3d direct, directly run these installation commands. Note, it
may be necessary to modify the Jax installation per the version of CUDA and
CuDNN. See: https://github.com/google/jax#pip-installation-gpu-cuda

```shell
pip install .
pip install --upgrade "jax3d[nesf]"
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
# Example alternative
# pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.5.3
```

# Datasets and Pre-Trained Checkpoints

Data can be copied from the Cloud bucket:
https://console.cloud.google.com/storage/browser/kubric-public/data/NeSFDatasets

Here is an example for the KLEVR Dataset:

```shell
wget https://storage.googleapis.com/kubric-public/data/NeSFDatasets/NeSF%20datasets/klevr.tar.gz
tar -xvf klevr.tar.gz
```

Pre-trained checkpoints are also availabe in the Cloud bucket.

Here is an example of copying and appropriately placing checkpoints for KLEVR
Dataset:

```shell
wget https://storage.googleapis.com/kubric-public/data/NeSFDatasets/NeRF%20checkpoints/klevr.tar.gz
mkdir klevr_checkpoints
mv klevr.tar.gz klevr_checkpoints
tar -xvf klevr.tar.gz
```

# Usage

To pretrain a NeRF model on a single scene, run the following. We recommend
parallelizing this operation across scenes. The result will be used in the next
step.

```shell
# Folder containing NeSF datasets, one folder per scene.
DATA_DIR=/path/to/your/dataset

# Any integer corresponding to a folder in $DATA_DIR
SCENE_IDX=0

# Where to write trained model checkpoints.
OUTPUT_DIR=/path/to/write/model/checkpoints

python3 -m jax3d.projects.nesf.nerfstatic.train \
  --gin_file="jax3d/jax3d/projects/nesf/nerfstatic/configs/public/nerf.gin" \
  --gin_bindings="DatasetParams.data_dir = '${DATA_DIR}'" \
  --gin_bindings="DatasetParams.train_scenes = '${SCENE_IDX}:$((${SCENE_IDX}+1))'" \
  --gin_bindings="TrainParams.train_dir = '${OUTPUT_DIR}/${SCENE_IDX}'" \
  --alsologtostderr
```

To evaluate a NeRF model on a single scene, run the following.

```shell
python3 -m jax3d.projects.nesf.nerfstatic.eval \
  --gin_file="jax3d/projects/nesf/nerfstatic/configs/public/nerf.gin" \
  --gin_bindings="DatasetParams.data_dir = '${DATA_DIR}'" \
  --gin_bindings="DatasetParams.train_scenes = '${SCENE_IDX}:$((${SCENE_IDX}+1))'" \
  --gin_bindings="TrainParams.train_dir = '${OUTPUT_DIR}/${SCENE_IDX}'" \
  --gin_bindings="EvalParams.sigma_grid_dir = '${OUTPUT_DIR}/sigma_grids'" \
  --alsologtostderr
```

To train the semantic module.

```shell

# Where to write the trained semantic model checkpoints.
OUTPUT_DIR_SEMANTIC=/path/to/write/semantic_model/checkpoints

# Where to find the trained NeRF model checkpoints.
NERF_MODEL_CKPT=$OUTPUT_DIR/sigma_grids/ OR /path/to/copied/pre-trained/checkpoints/klevr_checkpoints

python3 -m jax3d.projects.nesf.nerfstatic.train \
  --gin_file="jax3d/projects/nesf/nerfstatic/configs/public/nesf.gin" \
  --gin_bindings="DatasetParams.data_dir = '${DATA_DIR}'" \
  --gin_bindings="TrainParams.train_dir = '${OUTPUT_DIR_SEMANTIC}'" \
  --gin_bindings="TrainParams.nerf_model_ckpt = '${NERF_MODEL_CKPT}'" \
  --alsologtostderr
```

To evaluate the semantic module.

```shell
python3 -m jax3d.projects.nesf.nerfstatic.eval \
  --gin_file="jax3d/projects/nesf/nerfstatic/configs/public/nesf.gin" \
  --gin_bindings="DatasetParams.data_dir = '${DATA_DIR}'" \
  --gin_bindings="TrainParams.train_dir = '${OUTPUT_DIR_SEMANTIC}'" \
  --gin_bindings="TrainParams.nerf_model_ckpt = '${NERF_MODEL_CKPT}'" \
  --alsologtostderr
```
