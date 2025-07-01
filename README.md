# lightstereo_cpp

## About This Project

该项目是`LightStereo`算法的c++实现，包括`TensorRT`、`RKNN`、`OnnxRuntime`三种硬件平台(推理引擎)，并对原工程[OpenStereo/LightStereo](https://github.com/XiandaGuo/OpenStereo)导出onnx的代码进行了优化，提高其在`非nvidia`平台的性能。

## Features

1. 支持多种推理引擎: `TensorRT`、`RKNN`、`OnnxRuntime`
2. 支持异步、多核推理，算法吞吐量较高，特别是`RK3588`平台
3. 支持部署后模型的正确性、性能、精度测试。

## Demo

| <img src="./assets/left.png" alt="1" width="500"> | <img src="./assets/disp_color.png" alt="1" width="500"> |
|:----------------------------------------:|:----:|
| **left image**  | **disp in color** |

以下带有**opt**标志的代表在原工程[OpenStereo](https://github.com/XiandaGuo/OpenStereo)基础上，优化模型结构后导出的onnx模型，具体请查看[issue_link](https://github.com/XiandaGuo/OpenStereo/issues/212).

带有***async***标志的代表使用异步流程进行推理。

|  nvidia-3080-laptop   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  lightstereo(fp16) - origin   |   **388**   |  150%   |
|  lightstereo(fp16) - opt  |   370   |  150%   |
|  lightstereo(fp16) - origin - ***async***  |   **418**   |  170%   |
|  lightstereo(fp16) - opt - ***async***  |   390   |  170%   |


|  jetson-orin-nx-16GB   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  lightstereo(fp16) - origin   |   **70**   |  65%   |
|  lightstereo(fp16) - opt  |   65   |  70%   |
|  lightstereo(fp16) - origin - ***async***  |   **76**   |  80%   |
|  lightstereo(fp16) - opt - ***async***  |   69   |  85%   |


|  orangepi-5-plus-16GB   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  lightstereo(fp16) - origin   |   3.7   |  65%   |
|  lightstereo(fp16) - **opt**  |   **9**   |  **35%**   |
|  lightstereo(fp16) - origin - ***async***  |   14   |  210%   |
|  lightstereo(fp16) - **opt** - ***async***  |   **29**   |  **90%**   |

|  intel-i7-11800H   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  lightstereo(fp16) - origin   |   7   |  800%   |
|  lightstereo(fp16) - **opt**  |   **9**   |  800%   |

## Usage

### Download Project

下载git项目
```bash
git clone git@github.com:zz990099/lightstereo_cpp.git
cd lightstereo_cpp
git submodule init && git submodule update
```

### Build Enviroment

使用docker构建工作环境
```bash
cd lightstereo_cpp
bash easy_deploy_tool/docker/easy_deploy_startup.sh # 选择对应的平台和环境
bash easy_deploy_tool/docker/into_docker.sh
```

### Compile Codes

***支持stereo-matching算法的evaluation*** 

使用`-DENABLE_DEBUG_OUTPUT=ON`来开启测试log输出

在docker容器内，编译工程. 使用 `-DENABLE_*`宏来启用某种推理框架，可用的有: `-DENABLE_TENSORRT=ON`、`-DENABLE_RKNN=ON`、`-DENABLE_ORT=ON`，可以兼容。 
```bash
cd /workspace
mdkir build && cd build
cmake .. -DENABLE_DEBUG_OUTPUT=OFF \
         -DBUILD_TESTING=ON \
         -DBUILD_EVAL=ON \
         -DBUILD_BENCHMARK=ON \
         -DENABLE_TENSORRT=ON
make -j
```

### Convert Model

在docker容器内，运行模型转换脚本
```bash
cd /workspace
bash tools/cvt_onnx2trt.sh
# 或者运行python脚本，将模型转换为rknn
bash tools/cvt_onnx2rknn.sh
```

### Run Test Cases

运行测试用例，具体测试用例请参考代码。
```bash
cd /workspace/build
# 运行正确性测试
./bin/test_stereo_lightstereo
# 运行性能benchmark
./bin/benchmark_stereo_lightstereo
# 运行精度测试(epe)
./bin/eval_stereo_lightstereo
```

### Prepare Dataset for Evaluation

在[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)上下载`flyingthings3d_frames_finalpass`和`flyingthings3d_disparity`，解压后放到`/workspace/test_data/sceneflow/FlyingThings3D`下

## References

- [OpenStereo/LightStereo](https://github.com/XiandaGuo/OpenStereo)
- [EasyDeployTool](https://github.com/zz990099/EasyDeployTool)
