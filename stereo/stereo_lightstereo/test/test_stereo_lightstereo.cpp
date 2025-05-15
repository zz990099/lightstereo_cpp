#include <gtest/gtest.h>

#include "fps_counter.h"
#include "image_drawer.h"
#include "fs_util.h"
#include "detection_2d_util/detection_2d_util.h"
#include "stereo_lightstereo/lightstereo.hpp"

/**************************
****  trt core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;
using namespace stereo;

void lightstereo_test_correctness(const std::shared_ptr<BaseStereoMatchingModel> &lightstereo_model,
                                  const std::string                              &left_image_path,
                                  const std::string                              &right_image_path,
                                  const std::string                              &test_result_path)
{
  auto left  = cv::imread(left_image_path);
  auto right = cv::imread(right_image_path);
  CHECK(!left.empty() && !right.empty()) << "Failed to read images, path : \r\n"
                                         << left_image_path << "\r\n"
                                         << right_image_path;

  cv::Mat disp;
  CHECK(lightstereo_model->ComputeDisp(left, right, disp));

  if (!test_result_path.empty())
  {
    double minVal, maxVal;
    cv::minMaxLoc(disp, &minVal, &maxVal);
    cv::Mat normalized_disp_pred;
    disp.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                   -minVal * 255.0 / (maxVal - minVal));

    cv::Mat color_normalized_disp_pred;
    cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
    cv::imwrite(test_result_path, color_normalized_disp_pred);
  }
}

void lightstereo_test_async_correctness(
    const std::shared_ptr<BaseStereoMatchingModel> &lightstereo_model,
    const std::string                              &left_image_path,
    const std::string                              &right_image_path,
    const std::string                              &test_result_path)
{
  auto left  = cv::imread(left_image_path);
  auto right = cv::imread(right_image_path);
  CHECK(!left.empty() && !right.empty()) << "Failed to read images, path : \r\n"
                                         << left_image_path << "\r\n"
                                         << right_image_path;

  lightstereo_model->InitPipeline();

  auto async_func = [&]() { return lightstereo_model->ComputeDispAsync(left, right); };

  auto thread_fut = std::async(std::launch::async, async_func);

  auto stereo_fut = thread_fut.get();

  CHECK(stereo_fut.valid());

  cv::Mat disp = stereo_fut.get();

  if (!test_result_path.empty())
  {
    double minVal, maxVal;
    cv::minMaxLoc(disp, &minVal, &maxVal);
    cv::Mat normalized_disp_pred;
    disp.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                   -minVal * 255.0 / (maxVal - minVal));

    cv::Mat color_normalized_disp_pred;
    cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
    cv::imwrite(test_result_path, color_normalized_disp_pred);
  }
}

void lightstereo_test_speed(const std::shared_ptr<BaseStereoMatchingModel> &lightstereo_model,
                            const std::string                              &left_image_path,
                            const std::string                              &right_image_path,
                            size_t                                          predict_rounds)
{
  auto left  = cv::imread(left_image_path);
  auto right = cv::imread(right_image_path);
  CHECK(!left.empty() && !right.empty()) << "Failed to read images, path : \r\n"
                                         << left_image_path << "\r\n"
                                         << right_image_path;

  FPSCounter fps_counter;
  fps_counter.Start();
  for (int i = 0; i < predict_rounds; ++i)
  {
    cv::Mat disp;
    lightstereo_model->ComputeDisp(left, right, disp);
    fps_counter.Count(1);
    if (i % (predict_rounds / 10) == 0)
    {
      LOG(WARNING) << "Average qps: " << fps_counter.GetFPS();
    }
  }
}

void lightstereo_test_async_speed(const std::shared_ptr<BaseStereoMatchingModel> &lightstereo_model,
                                  const std::string                              &left_image_path,
                                  const std::string                              &right_image_path,
                                  size_t                                          predict_rounds)
{
  auto left  = cv::imread(left_image_path);
  auto right = cv::imread(right_image_path);
  CHECK(!left.empty() && !right.empty()) << "Failed to read images, path : \r\n"
                                         << left_image_path << "\r\n"
                                         << right_image_path;

  lightstereo_model->InitPipeline();

  deploy_core::BlockQueue<std::shared_ptr<std::future<cv::Mat>>> future_bq(100);

  auto func_push_data = [&]() {
    int index = 0;
    while (index++ < predict_rounds)
    {
      auto p_fut = std::make_shared<std::future<cv::Mat>>(
          lightstereo_model->ComputeDispAsync(left.clone(), right.clone()));
      future_bq.BlockPush(p_fut);
    }
    future_bq.SetNoMoreInput();
  };

  FPSCounter fps_counter;
  auto       func_take_results = [&]() {
    int index = 0;
    fps_counter.Start();
    while (true)
    {
      auto output = future_bq.Take();
      if (!output.has_value())
        break;
      output.value()->get();
      fps_counter.Count(1);
      if ((index++) % (predict_rounds / 10) == 0)
      {
        LOG(WARNING) << "Average qps: " << fps_counter.GetFPS();
      }
    }
  };

  std::thread t_push(func_push_data);
  std::thread t_take(func_take_results);

  t_push.join();
  lightstereo_model->StopPipeline();
  t_take.join();
  lightstereo_model->ClosePipeline();

  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
}

#define GEN_TEST_CASES(Tag, FixtureClass)                                                       \
  TEST_F(FixtureClass, test_lightstereo_##Tag##_correctness)                                    \
  {                                                                                             \
    lightstereo_test_correctness(lightstereo_model_, left_image_path_, right_image_path_,       \
                                 test_lightstereo_result_path_);                                \
  }                                                                                             \
  TEST_F(FixtureClass, test_lightstereo_##Tag##_async_correctness)                              \
  {                                                                                             \
    lightstereo_test_async_correctness(lightstereo_model_, left_image_path_, right_image_path_, \
                                       test_lightstereo_result_path_);                          \
  }                                                                                             \
  TEST_F(FixtureClass, test_lightstereo_##Tag##_speed)                                          \
  {                                                                                             \
    lightstereo_test_speed(lightstereo_model_, left_image_path_, right_image_path_,             \
                           speed_test_predict_rounds_);                                         \
  }                                                                                             \
  TEST_F(FixtureClass, test_lightstereo_##Tag##_async_speed)                                    \
  {                                                                                             \
    lightstereo_test_async_speed(lightstereo_model_, left_image_path_, right_image_path_,       \
                                 speed_test_predict_rounds_);                                   \
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

#include "trt_core/trt_core.h"

class Lightstereo_TensorRT_Fixture : public BaselightstereoFixture {
public:
  void SetUp() override
  {
    auto engine =
        CreateTrtInferCore("/workspace/models/lightstereo_s_sceneflow_general_opt.engine");
    auto preprocess_block = CreateCudaDetPreProcess();
    lightstereo_model_    = stereo::CreateLightStereoModel(engine, preprocess_block, 256, 512,
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

#include "ort_core/ort_core.h"

class Lightstereo_OnnxRuntime_Fixture : public BaselightstereoFixture {
public:
  void SetUp() override
  {
    auto engine = CreateOrtInferCore("/workspace/models/lightstereo_s_sceneflow_general_opt.onnx");
    auto preprocess_block =
        CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);
    lightstereo_model_ = stereo::CreateLightStereoModel(engine, preprocess_block, 256, 512,
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

#include "rknn_core/rknn_core.h"

class Lightstereo_Rknn_Fixture : public BaselightstereoFixture {
public:
  void SetUp() override
  {
    auto engine = CreateRknnInferCore(
        "/workspace/models/lightstereo_s_sceneflow_general_opt.rknn",
        {{"left_img", RknnInputTensorType::RK_UINT8}, {"right_img", RknnInputTensorType::RK_UINT8}},
        5, 3);
    auto preprocess_block      = CreateCpuDetPreProcess({}, {}, false, false);
    lightstereo_model_         = stereo::CreateLightStereoModel(engine, preprocess_block, 256, 512,
                                                                {"left_img", "right_img"}, {"disp_pred"});
    speed_test_predict_rounds_ = 200;
    left_image_path_           = "/workspace/test_data/left.png";
    right_image_path_          = "/workspace/test_data/right.png";
    test_lightstereo_result_path_ = "/workspace/test_data/lightstereo_rknn_test_result.png";
  }
};

GEN_TEST_CASES(rknn, Lightstereo_Rknn_Fixture);

#endif
