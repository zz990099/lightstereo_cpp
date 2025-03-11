#!/bin/bash

echo "Converting LightStereo onnx model to tensorrt ..."
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/lightstereo_s_sceneflow_general_origin.onnx \
                              --saveEngine=/workspace/models/lightstereo_s_sceneflow_general.engine \
                              --fp16