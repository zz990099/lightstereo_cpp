#include <gtest/gtest.h>

#include "detection_2d_util/detection_2d_util.hpp"
#include "stereo_lightstereo/lightstereo.hpp"
#include "test_utils/stereo_matching_test_utils.hpp"

using namespace easy_deploy;

#define GEN_TEST_CASES(Tag, FixtureClass)                                                         \
  TEST_F(FixtureClass, test_lightstereo_##Tag##_correctness)                                      \
  {                                                                                               \
    test_stereo_matching_algorithm_correctness(lightstereo_model_, left_image_path_,              \
                                               right_image_path_, test_lightstereo_result_path_); \
  }                                                                                               \
  TEST_F(FixtureClass, test_lightstereo_##Tag##_async_correctness)                                \
  {                                                                                               \
    test_stereo_matching_algorithm_async_correctness(                                             \
        lightstereo_model_, left_image_path_, right_image_path_, test_lightstereo_result_path_);  \
  }

class BaselightstereoFixture : public testing::Test {
protected:
  std::shared_ptr<BaseStereoMatchingModel> lightstereo_model_;

  std::string left_image_path_;
  std::string right_image_path_;
  std::string test_lightstereo_result_path_;
  size_t      speed_test_predict_rounds_;
};

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

class Lightstereo_TensorRT_Fixture : public BaselightstereoFixture {
public:
  void SetUp() override
  {
    auto engine =
        CreateTrtInferCore("/workspace/models/lightstereo_s_sceneflow_general_opt.engine");
    auto preprocess_block = CreateCudaDetPreProcess();
    lightstereo_model_    = CreateLightStereoModel(engine, preprocess_block, 256, 512,
                                                   {"left_img", "right_img"}, {"disp_pred"});

    speed_test_predict_rounds_    = 2000;
    left_image_path_              = "/workspace/test_data/left.png";
    right_image_path_             = "/workspace/test_data/right.png";
    test_lightstereo_result_path_ = "/workspace/test_data/lightstereo_trt_test_result.png";
  }
};

GEN_TEST_CASES(tensorrt, Lightstereo_TensorRT_Fixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

class Lightstereo_OnnxRuntime_Fixture : public BaselightstereoFixture {
public:
  void SetUp() override
  {
    auto engine =
        CreateOrtInferCore("/workspace/models/lightstereo_s_sceneflow_general_opt.onnx",
                           {{"left_img", {1, 3, 256, 512}}, {"right_img", {1, 3, 256, 512}}},
                           {{"disp_pred", {1, 1, 256, 512}}});
    auto preprocess_block =
        CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);
    lightstereo_model_ = CreateLightStereoModel(engine, preprocess_block, 256, 512,
                                                {"left_img", "right_img"}, {"disp_pred"});

    speed_test_predict_rounds_    = 100;
    left_image_path_              = "/workspace/test_data/left.png";
    right_image_path_             = "/workspace/test_data/right.png";
    test_lightstereo_result_path_ = "/workspace/test_data/lightstereo_ort_test_result.png";
  }
};

GEN_TEST_CASES(onnxruntime, Lightstereo_OnnxRuntime_Fixture);

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.hpp"

class Lightstereo_Rknn_Fixture : public BaselightstereoFixture {
public:
  void SetUp() override
  {
    auto engine = CreateRknnInferCore(
        "/workspace/models/lightstereo_s_sceneflow_general_opt.rknn",
        {{"left_img", RknnInputTensorType::RK_UINT8}, {"right_img", RknnInputTensorType::RK_UINT8}},
        5, 3);
    auto preprocess_block         = CreateCpuDetPreProcess({}, {}, false, false);
    lightstereo_model_            = CreateLightStereoModel(engine, preprocess_block, 256, 512,
                                                           {"left_img", "right_img"}, {"disp_pred"});
    speed_test_predict_rounds_    = 200;
    left_image_path_              = "/workspace/test_data/left.png";
    right_image_path_             = "/workspace/test_data/right.png";
    test_lightstereo_result_path_ = "/workspace/test_data/lightstereo_rknn_test_result.png";
  }
};

GEN_TEST_CASES(rknn, Lightstereo_Rknn_Fixture);

#endif
