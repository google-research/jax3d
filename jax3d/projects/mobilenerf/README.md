# MobileNeRF

This repository contains the source code for the paper MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Eﬀicient Neural Field Rendering on Mobile Architectures.

This code is created by [Zhiqin Chen](https://czq142857.github.io/) when he was a student researcher at Google.

*Please note that this is not an officially supported Google product.*


## Installation

You will need 8 v100 GPUs to successfully train the model.

We recommend using [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. Clone the repo, go to the mobilenerf folder, and run the following commands:

```
conda create --name mobilenerf python=3.6.13; conda activate mobilenerf
conda install pip; pip install --upgrade pip
pip install -r requirements.txt
```

Please make sure that your jax supports GPU. You might need to re-install jax by following the [jax installation guide](https://github.com/google/jax#installation).



## Data

Please download the datasets from the [NeRF official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Please download and unzip `nerf_synthetic.zip` and `nerf_llff_data.zip`.

**(TODO: how to download unbounded scenes from Mip-NeRF 360?)**

## Training

The training code is in three .py files, corresponding to the three training stages: 1. continuous opacity, 2. binarization and supersampling, 3. extracting meshes and textures.

First, modify the parameters in all .py files:
```
scene_type = "synthetic"
object_name = "chair"
scene_dir = "datasets/nerf_synthetic/"+object_name
```
*scene_type* can be synthetic, forwardfacing, or real360. *object_name* is the name of the scene to be trained; the available names are listed in the code. *scene_dir* is the folder holding the training data.

Afterwards, run the three .py files consecutively
```
python stage1.py
python stage2.py
python stage3.py
```
The intermediate weights will be saved in folder *weights* and the intermediate outputs (sample testing images, loss curves) will be written to folder *samples*. The output meshes+textures will be saved in folder *obj_phone*.

Note: the stage-1 training could occasionally fail for unknown reasons (especially on the bicycle scene); switch to a different set of GPUs could solve this.

Note: For unbounded 360 degree scenes, ```stage3.py``` will only extract meshes of the center unit cube. To extract the entire scene, use ```python stage3_with_box.py```.

It takes 8 hours to train the first stage, 12-16 hours to train the second, and 1-3 hours to run the third.

## Running the viewer

The viewer code is provided in this repo, as three .html files for three types of scenes.

You can set up a local server on your machine, e.g.,
```
cd folder_containing_the_html
python -m http.server
```
Then open
```
localhost:8000/view_synthetic.html?obj=chair
```
Note that you should put the meshes+textures of the chair model in a folder *chair_phone*. The folder should be in the same directory as the html file.

Please allow some time for the scenes to load. Use left mouse button to rotate, right mouse button to pan (especially for forward-facing scenes), and scroll wheel to zoom. On phones, Use you fingers to rotate or pan or zoom. Resize the window (or landscape<->portrait your phone) to show the resolution.