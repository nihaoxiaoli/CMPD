# CMPD
Confidence-aware Fusion using Dempster-Shafer Theory for Multispectral Pedestrian Detection

# Comparison Results
You can directly click on the [link](https://pan.baidu.com/s/1mVwdsYu0y3J-I5Pgi-GH4w?pwd=c7sb) to download our results for drawing the FPPI-MR curve.

# Demo
## 1. Dependencies
You can refer to the environment of MBNet. [link](https://github.com/CalayZhou/MBNet)

or 

You can follow the steps below to configure the environment.

 
 >make sure the GPU enviroment is the same as above (cuda10.0,cudnn7.6), otherwise you may have to compile the `nms` and `utils` according to https://github.com/endernewton/tf-faster-rcnn. Besides, check the keras version is keras2.1, i find there may be some mistakes if the keras version is higher. To be as simple as possible, I recommend installing the dependencies with Anaconda as follows:
 
 ```
1. conda create -n python36 python=3.6
2. conda activate python36
3. conda install cudatoolkit=10.0
4. conda install cudnn=7.6
5. conda install tensorflow-gpu=1.14
6. conda install keras=2.1
7. conda install opencv
8. python demo.py
 ```


## 2. Demo example

Ensure the composition of the folder is as shown in the figure.
```
+--data
|   +-- kaist_test
|   |   +-- kaist_test_visible
|   |   +-- kaist_test_lwir
+--output
|   +-- resnet_e7_l280.hdf5
+--framework
+--README.md
```

The specific operation is as follows.
> 1. Check the [CMPD model](https://pan.baidu.com/s/1guvScGBwgKfNqX-CyLfm7A?pwd=2dhw) is available at ./output/resnet_e7_l280.hdf5
> 2. Enter folder 'framework': `cd framework`
> 2. Run the script: `python demo.py`
> 3. The detection result is saved at ./result/. (This folder will be created automatically. )



# Acknowledgements
This pipeline is largely built on [MBNet](https://github.com/CalayZhou/MBNet). Thank for this great work. Meanwhile, if you encounter any problems during the configuration process, you can check [issue](https://github.com/CalayZhou/MBNet/issues) of MBNet to see if you can find answers there.

