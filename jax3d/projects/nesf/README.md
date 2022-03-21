# NeSF

NeSF fork of jax3d. This codebase contains the original implementation of neural
semantic fields, a method for 3D semantic segmentation from posed 2D inputs and
supervision. Learn more at https://nesf3d.github.io/.

# Installation

NeSF requires additional libraries other than those required for `jax3d`. The
following installs those libraries and support for single- and multi-GPU
training.

```shell
pip install --upgrade "jax3d[nesf]"
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
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

git clone https://github.com/google-research/jax3d.git
python -m jax3d.projects.nesf.nerfstatic.train \
  --gin_file="jax3d/jax3d/projects/nesf/nerfstatic/configs/public/nerf.gin" \
  --gin_bindings="DatasetParams.data_dir = '${DATA_DIR}'" \
  --gin_bindings="DatasetParams.train_scenes = '${SCENE_IDX}:$((${SCENE_IDX}+1))'" \
  --gin_bindings="TrainParams.train_dir = '${OUTPUT_DIR}/${SCENE_IDX}'" \
  --alsologtostderr
```
