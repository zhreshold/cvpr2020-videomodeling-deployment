# cvpr2020-videomodeling-deployment
Materials for demonstrating video model deployment

## Prerequisites
To be able to run these jupyter notebooks, you will need to install `mxnet`, `gluoncv` and `tvm`(for third notebook only).

```
pip install mxnet gluoncv decord jupyter
```

For TVM installation, please check out [tvm](https://tvm.apache.org/docs/install/index.html).


## How to build the Jetson Demo App

(This tutorial is verified on JetPack 4.4).


Install the system packages
```
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-setuptools make cmake git
sudo apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
```

Make sure you have cloned the repo recursively with the submodules
```
git submodule update --recursive --init
```

Build the demo app
```
cd path_to_this_repo/tvm_deploy
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

Now the `video_classification` app is ready to go!

## How to use the Jetson Demo App

First of all, make sure you have played with `03_deploy_video_model_to_tvm.ipynb` and have exported tvm runtime lib `xxx_deploy_lib.so`, `xxx_deploy_graph.json`, `xxx_deploy_0000.params`, and `xxx_synset.txt`.
To execute the app, copy the executable `video_classification` to the same directory with the parameter files.

Then 
```bash
./video_classification test.mkv model_name --gpu gpu_id
```

For example
```
./video_classification pancake.mkv resnet18_v1b_kinetics400 --gpu 0
```

Outputs:
```
[13:27:08] /home/xavier/cvpr20-tutorial/cvpr2020-videomodeling-deployment/tvm_deploy/src/classification.cpp:116: Read 13 frames.
[13:27:08] /home/xavier/cvpr20-tutorial/cvpr2020-videomodeling-deployment/tvm_deploy/src/classification.cpp:147: Elapsed time {Forward->Result}: 143.906 ms
[13:27:08] /home/xavier/cvpr20-tutorial/cvpr2020-videomodeling-deployment/tvm_deploy/src/classification.cpp:161: The input picture is classified to be
        [flipping_pancake], with probability 0.996
        [playing_drums], with probability 0.003
        [air_drumming], with probability 0.000
        [playing_cymbals], with probability 0.000
        [cooking_chicken], with probability 0.000

```