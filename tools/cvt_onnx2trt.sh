#!/bin/bash

echo "Converting LightStereo onnx model to tensorrt ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/lightstereo_s_sceneflow_general_opt_256_512.onnx \
                              --saveEngine=/workspace/models/lightstereo_s_sceneflow_general_opt_256_512.engine \
                              --fp16

/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/lightstereo_s_sceneflow_general_opt_576_960.onnx \
                              --saveEngine=/workspace/models/lightstereo_s_sceneflow_general_opt_576_960.engine \
                              --fp16