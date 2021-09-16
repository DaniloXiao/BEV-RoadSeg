# BEV-RoadSeg
BEV-RoadSeg for Freespace Detection in PyTorch.

## Introduction
This is a repository based on `SNE-Roadseg`, including` Python onnx and tensorRT API` versions. For source code and paper, see: https://github.com/hlwang1124/SNE-RoadSeg.

In this repo, we provide the training and testing setup for the `Ouster-OS1-128 Lidar Road Dataset`, you can replace it with KITTI Dataset.

<p align="center">
<img src="doc/roadseg.gif" width="60%"/>
</p>


## Setup
Please setup the Road Dataset and pretrained weights according to the following folder structure:
```
BEV-RoadSeg
 |-- checkpoints
 |  |-- kitti
 |  |  |-- kitti_net_RoadSeg.pth
 |-- data
 |-- datasets
 |  |-- kitti
 |  |  |-- training
 |  |  |  |-- gt_image_2
 |  |  |  |-- image_2
 |  |  |-- validation
 |  |  |  |-- gt_image_2
 |  |  |  |-- image_2
 |  |  |-- testing
 |  |  |  |-- depth_u16
 |  |  |  |-- image_2
 |  |  |-- image_cam #Creates datas for merge_cam_to_bev
 |  |  |-- velodyne #Creates dataset for bev
 ...
```
The pretrained weights `kitti_net_RoadSeg.pth` for our RoadSeg-18 can be downloaded from ` GoogleDrive` [here](https://drive.google.com/file/d/1t6qj8O5gEq2Ij7pdXZ5UNQ2Vu3IB1FNb/view?usp=sharing),
` BaiduDrive(code:da16)` [here](https://pan.baidu.com/s/1oaHtXETrvf-rZ0YmVSdhWw).

## Usage

### Data preparation
For `.pkl` Lidar Dataset, you need to setup `datasets/kitti/velodyne`folder as mentioned above.
```
python3 lidar_pkl_to_bev.py
```
and you will get the `img_bev` and `img_cam` results in `datasets/kitti/velodyne`. The segmentation annotation is not provided, hence we need to label drivable aera from img_bev.


### Testing on the Road dataset
You need to setup the `checkpoints` and the `datasets/kitti/testing` folder as mentioned above. Then, run the following script:
```
bash ./scripts/test.sh
```
and you will get the prediction results in `testresults`.

### Detect on the Road dataset
You need to setup the `checkpoints` and the `datasets/kitti/testing` folder as mentioned above. Then, run the following script:
```
bash ./scripts/detect.sh
```
and you will get the video or img results in `testresults`, `merge img_cam with img_bev`.

### Training on the Road dataset
For training, you need to setup the `datasets/kitti` folder as mentioned above. You can split the original training set into a new training set and a validation set as you like. Then, run the following script:
```
bash ./scripts/train.sh
```
and the weights will be saved in `checkpoints` and the `tensorboard` record containing the loss curves as well as the performance on the validation set will be save in `runs`.


### Build ONNX
.pth to .onnx:
```
python3 export_onnx.py
```

### Build trt engine
We support many different types of engine export, such as static `fp32, fp16, and int8 quantization` :
fp32, fp16:
```
python3 tensorRT_bulid_engine.py  --onnx_path ./checkpoints/kitti/kitti_net_RoadSeg.onnx --mode fp16
```
int8:
```
python3 tensorRT_bulid_engine.py  --onnx_path ./checkpoints/kitti/kitti_net_RoadSeg.onnx --mode int8 --int8_calibration_path ./datasets/kitti/training/image_2/
```

### TensorRT detect on the Road dataset
You need to setup the `checkpoints` and the `datasets/kitti/testing` folder as mentioned above. Then, run the following script:
```
bash ./scripts/tensorRT_detect.sh
```
and you will get the video or img results in `testresults`, `merge img_cam with img_bev`.


