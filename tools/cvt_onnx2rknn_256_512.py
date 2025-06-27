import sys
from rknn.api import RKNN
# onnx model path
onnx_model_path = '/workspace/models/lightstereo_s_sceneflow_general_opt_256_512.onnx'
# quant data
DATASET_PATH = '/workspace/test_data/quant_data/dataset.txt'
# output paths
DEFAULT_RKNN_PATH = '/workspace/models/lightstereo_s_sceneflow_general_opt_256_512.rknn'

if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[123.675, 116.28, 103.53], [123.675, 116.28, 103.53]], 
                std_values=[[58.395, 57.12, 57.375], [58.395, 57.12, 57.375]], 
                target_platform="rk3588",
                optimization_level=2)

    ret = rknn.load_onnx(model=onnx_model_path)
    ret = rknn.build(do_quantization=False, dataset=DATASET_PATH)

    ret = rknn.export_rknn(DEFAULT_RKNN_PATH)
    print('done')

    # Release
    rknn.release()