# Tracking through Containers and Occluders in the Wild

Basile Van Hoorick, Pavel Tokmakov, Simon Stent, Jie Li, Carl Vondrick

Columbia University, Toyota Research Institute, Woven Planet

Published in CVPR 2023

[Paper](https://tcow.cs.columbia.edu/TCOW_v3.pdf) | [Website](https://tcow.cs.columbia.edu/) | [Results](https://tcow.cs.columbia.edu/#results) | [Datasets](https://tcow.cs.columbia.edu/#datasets) | [Models](https://github.com/basilevh/tcow#models)

https://user-images.githubusercontent.com/18504625/236341510-765ae45e-1704-44cf-9b7c-9d78d261297b.mp4

This repository contains the Python code published as part of our paper _"[Tracking through Containers and Occluders in the Wild](https://tcow.cs.columbia.edu/TCOW_v3.pdf)"_ (abbreviated **TCOW**).

## Setup

We recommend setting up a virtual environment as follows:

```bash
conda create -n tcow python=3.9
conda activate tcow
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

We use a modified version of Meta's [TimeSformer](https://github.com/facebookresearch/TimeSformer) library, which can be installed like this:

```bash
cd third_party/TimeSformer
pip install -e .
```

At the time of writing, the latest version of PyTorch is 2.0.1, which is what the above commands will install. Although our method was developed with PyTorch 1.13.1, we have not encountered any issues with the latest version. If you do encounter issues, please feel free to us know.

## Training

First, download the Kubric Random training set [here](https://tcow.cs.columbia.edu/#datasets).

Next, here is the main command that we used to train our model:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --name v1 --data_path /path/to/datasets/kubric_random/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --frame_height 240 --frame_width 320 --causal_attention 1 --seeker_query_time 0
```

* Since the assigned name of this experiment/run is `v1`, checkpointing will happen in `checkpoints/v1 `, and logs and visualizations will be stored in `logs/v1 `. Note that there are more options and hyperparameters available in `args.py`. Feel free to play around with those if you want to adapt our model to your research and/or experimentation purposes.

* If you wish to use different datasets for training, you will have to write your own dataloader, since we implemented a custom dataloader especially for Kubric Random (see `data/data_kubric.py`).

* The example command uses the first 2 GPUs and a batch size of 1 per GPU, but uses around 38 GB of VRAM due to the memory complexity characteristics of the transformer architecture. We used NVIDIA RTX A6000 in our experiments, but try decreasing the spatial and/or temporal resolution if you run out of memory.

## Evaluation

For testing, keep in mind that we have to apply some tricks to accommodate input videos with arbitrary durations. Our model accepts a fixed number of T=30 frames as input, so we typically temporally subsample videos with a certain integer stride value. In the interest of thoroughness, we typically use multiple strides (each of which correspond to clips of different difficulties, since more things tend to happen in longer videos) and subsequently average the results.

Download the Kubric and/or Rubric test sets of your choice [here](https://tcow.cs.columbia.edu/#datasets), and extract them to `datasets/` (or use a symbolic link). The contents of `datasets/` should look similar to this (you might need to rename some things):

```
datasets/
  kubric_random/
    train/
      kubcon_v10_scn01234/
      ...
    val/
    test/
  kubric_containers/
    kubbench_v3_scn012_box_push_container_slide/
    ...
  rubric/
    2_teaduck/
    3_mugduck/
    ...
  rubric_all_videos.txt
  rubric_office_videos.txt
  rubric_cupgames_videos.txt
  rubric_davytb_videos.txt
```

After training has finished, here is the command to generate all test results:

```bash
# Kubric Random test set
python eval/test.py --resume v1 --name v1_kr --gpu_id 0 --data_path datasets/kubric_random/ --num_queries 4 --extra_visuals 1

# Kubric Containers benchmark
python eval/test.py --resume v1 --name v1_kc --gpu_id 0 --data_path datasets/kubric_containers/ --num_queries 1 --extra_visuals 1

# Rubric benchmark (all videos)
python eval/test.py --resume v1 --name v1_ra --gpu_id 0 --data_path datasets/rubric_all_videos.txt --num_queries 1 --extra_visuals 1
```

The raw results are then stored in, for example, `logs/v1/test_v1_kc` for Kubric Containers. You can also use an already trained checkpoint, such as the one below (in which case you simply replace `v1` with `tcow`). Custom benchmarks are supported as well; simply follow the structure of videos and annotation file naming patterns in the Rubric folders.

Next, we select the representative outputs (to avoid redundancy) and average them to obtain final numbers (separately per dataset) as follows:

```bash
python eval/pick_represent.py --testres_path logs/v1/test_* --represent_guide rep_lists/*.txt --output_dir logs_rep/
```

This script will match the generated results with the appropriate categories and entries in the provided guides (see text files in `rep_lists/`), and store the final numbers in subfolders in `logs_rep/`. Open `_autosmr_i.csv` for an overview. We typically use the _weighted_ (not the _unweighted_) metrics for reporting, in order to average over the number of frames rather than the number of videos.

## Pretrained Models

We provide a checkpoint [here](https://tcow.cs.columbia.edu/models/tcow_pretrained.zip) for the main non-ablated TCOW network. Download and extract this archive in the root of the repository, such that you end up with:

```
tcow/
  checkpoints/
    tcow/
      checkpoint.pth
      checkpoint_epoch.txt
      checkpoint_name.txt
  logs/
    tcow/
      args_train.txt
```

The name of this experiment is `tcow`, so any test command above can be executed as long as you provide `--resume tcow`.

The provided pretrained model operates at a spatial resolution of 320 x 240, so an aspect ratio of 4:3. If any input video has a different aspect ratio, the evaluation code will simply apply a center crop first.

## Usage (Inference)

We did our best to support running our model in a plug-and-play fashion with arbitrary clips out of the box.

First, follow the instructions above to download and extract the model checkpoint.

Then, run the following example command to let TCOW (our model) track a toy duck behind a tea box:

```bash
python eval/test.py --resume tcow --name tcow_demo --gpu_id 0 --data_path demo/teaduck2.mp4 --num_queries 1 --extra_visuals 1
```

Since we pointed the script to the video `demo/teaduck2.mp4`, it will automatically find the query mask located at `demo/teaduck2_15_query.png ` and start tracking that object. The output segmentations will be produced and saved in `logs/tcow/test_tcow_demo/visuals/`.

In general, in order to prepare a custom video clip with a labeled query mask, make sure the file structure looks like this:

```
my_data/
  my_video.mp4
  my_video_0_query.png
  ...
```

Here, `my_video_0_query.png` must an image with the same resolution as the video file, where black pixels represent the background and a non-white color denotes the target object that you want to track. Note that you can change the frame index `0` in the file name to any other 0-based integer value if you want to start tracking something later in the video. Then, simply run the following command:

```bash
python eval/test.py --resume tcow --name tcow_p1 --gpu_id 0 --data_path my_data/my_video.mp4 --num_queries 1 --extra_visuals 1
```

This will automatically pick up the available query object mask files and run the model playing back the video at different speeds. There also exists a way of providing ground truth masks for instances and/or occluders and/or containers; we recommend looking at the file structure of the Rubric benchmark if you want to do this.

You can also point `--data_path` to a `.txt` file containing a list of (absolute or relative) paths of video files, and then the test script will iterate over all of them (this is in fact how Rubric evaluation is done).

## Dataset Generation

If you wish to generate your own synthetic data with Google's Kubric simulator, the virtual environment setup has to be a little bit more specific because of bpy. I got this working only on Linux with Python 3.7 and bpy for Blender 2.91 alpha.

#### _NOTE: I have not thoroughly vetted these setup instructions yet, because it is not trivial to do, so this part is not final. Please refer to the issues page if you run into problems._

```bash
conda create -n kubric python=3.7
conda activate kubric
wget https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
pip install bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
bpy_post_install
pip install kubric
pip install -r requirements.txt
```

Our Kubric scene templates are based on [MOVi-F](https://github.com/google-research/kubric/tree/main/challenges/movi#movi-f), but with quite a few changes (mainly X-ray annotations and altered object statistics) in order to be better suited to our task and setting in this paper.

For Kubric Random, modify the paths and options in `gen_kubric/export_kub_rand.py` and then run the following command:

```bash
python gen_kubric/export_kub_rand.py
```

For Kubric Containers, modify and run `gen_kubric/export_kub_cont.py` instead.

The job can occasionally crash for a variety of reasons, so I recommend surrounding it with a bash for loop. Another tip is to clear `/tmp` regularly.

## BibTeX Citation

If you utilize our code and/or dataset, please consider citing our paper:

```
@inproceedings{vanhoorick2023tcow,
title={Tracking through Containers and Occluders in the Wild},
author={Van Hoorick, Basile and Tokmakov, Pavel and Stent, Simon and Li, Jie and Vondrick, Carl},
journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2023}}
```
