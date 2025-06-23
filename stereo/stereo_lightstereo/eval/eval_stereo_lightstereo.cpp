#include "detection_2d_util/detection_2d_util.hpp"
#include "stereo_lightstereo/lightstereo.hpp"
#include "eval_utils/stereo_matching_eval_utils.hpp"

using namespace easy_deploy;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

class EvalAccuracyLightStereoTensorRTFixture : public EvalAccuracyStereoMatchingFixture {
public:
  SetUpReturnType SetUp() override
  {
    auto engine =
        CreateTrtInferCore("/workspace/models/lightstereo_s_sceneflow_general_opt.engine");
    auto preprocess_block = CreateCudaDetPreProcess();

    auto model = CreateLightStereoModel(engine, preprocess_block, 256, 512,
                                        {"left_img", "right_img"}, {"disp_pred"});

    const std::string sceneflow_val_txt_path =
        "/workspace/test_data/sceneflow/sceneflow_finalpass_test.txt";
    return {model, sceneflow_val_txt_path};
  }
};

RegisterEvalAccuracyStereoMatching(EvalAccuracyLightStereoTensorRTFixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

class EvalAccuracyLightStereoOnnxRuntimeFixture : public EvalAccuracyStereoMatchingFixture {
public:
  SetUpReturnType SetUp() override
  {
    auto engine =
        CreateOrtInferCore("/workspace/models/lightstereo_s_sceneflow_general_opt.onnx",
                           {{"left_img", {1, 3, 256, 512}}, {"right_img", {1, 3, 256, 512}}},
                           {{"disp_pred", {1, 1, 256, 512}}});
    auto preprocess_block =
        CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);

    auto model = CreateLightStereoModel(engine, preprocess_block, 256, 512,
                                        {"left_img", "right_img"}, {"disp_pred"});

    const std::string sceneflow_val_txt_path =
        "/workspace/test_data/sceneflow/sceneflow_finalpass_test.txt";
    return {model, sceneflow_val_txt_path};
  }
};

RegisterEvalAccuracyStereoMatching(EvalAccuracyLightStereoOnnxRuntimeFixture);

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.hpp"

class EvalAccuracyLightStereoRknnFixture : public EvalAccuracyStereoMatchingFixture {
public:
  SetUpReturnType SetUp() override
  {
    auto engine = CreateRknnInferCore(
        "/workspace/models/lightstereo_s_sceneflow_general_opt.rknn",
        {{"left_img", RknnInputTensorType::RK_UINT8}, {"right_img", RknnInputTensorType::RK_UINT8}},
        5, 3);
    auto preprocess_block = CreateCpuDetPreProcess({}, {}, false, false);

    auto model = CreateLightStereoModel(engine, preprocess_block, 256, 512,
                                        {"left_img", "right_img"}, {"disp_pred"});

    const std::string sceneflow_val_txt_path =
        "/workspace/test_data/sceneflow/sceneflow_finalpass_test.txt";
    return {model, sceneflow_val_txt_path};
  }
};

RegisterEvalAccuracyStereoMatching(EvalAccuracyLightStereoRknnFixture);

#endif

EVAL_MAIN()
