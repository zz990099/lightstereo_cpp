#include <gtest/gtest.h>

#include "rknn_core/rknn_core.h"
#include "tests/fps_counter.h"
#include "detection_2d_util/detection_2d_util.h"
#include "stereo_lightstereo/lightstereo.hpp"

/**************************
****  rknn core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;
using namespace stereo;

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoModel()
{
  auto engine = CreateRknnInferCore(
      "/workspace/models/lightstereo_s_sceneflow_general.rknn",
      {{"left_img", RknnInputTensorType::RK_UINT8}, {"right_img", RknnInputTensorType::RK_UINT8}}, 5, 3);
  auto preprocess_block = CreateCpuDetPreProcess({}, {}, false, false);
  auto model            = stereo::CreateLightStereoModel(engine, preprocess_block, 256, 512,
                                                         {"left_img", "right_img"}, {"disp_pred"});

  return model;
}

std::tuple<cv::Mat, cv::Mat> ReadTestImages()
{
  auto left  = cv::imread("/workspace/test_data/left.png");
  auto right = cv::imread("/workspace/test_data/right.png");

  return {left, right};
}

TEST(lightstereo_test, rknn_core_correctness)
{
  auto model         = CreateLightStereoModel();
  auto [left, right] = ReadTestImages();

  cv::Mat disp;
  model->ComputeDisp(left, right, disp);

  double minVal, maxVal;
  cv::minMaxLoc(disp, &minVal, &maxVal);
  cv::Mat normalized_disp_pred;
  disp.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                 -minVal * 255.0 / (maxVal - minVal));

  cv::Mat color_normalized_disp_pred;
  cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
  cv::imwrite("/workspace/test_data/lightstereo_result_color.png", color_normalized_disp_pred);
}

TEST(lightstereo_test, rknn_core_speed)
{
  auto model         = CreateLightStereoModel();
  auto [left, right] = ReadTestImages();

  FPSCounter fps_counter;
  fps_counter.Start();
  for (int i = 0; i < 1000; ++i)
  {
    cv::Mat disp;
    model->ComputeDisp(left, right, disp);
    fps_counter.Count(1);
    if (i % 100 == 0)
    {
      LOG(WARNING) << "cur fps : " << fps_counter.GetFPS();
    }
  }
}



TEST(lightstereo_test, rknn_core_pipeline_correctness)
{
  auto model         = CreateLightStereoModel();
  model->InitPipeline();
  auto [left, right] = ReadTestImages();

  auto async_func = [&]() {
    return model->ComputeDispAsync(left, right);
  };

  auto thread_fut = std::async(std::launch::async, async_func);

  auto stereo_fut = thread_fut.get();

  CHECK(stereo_fut.valid());

  cv::Mat disp = stereo_fut.get();

  double minVal, maxVal;
  cv::minMaxLoc(disp, &minVal, &maxVal);
  cv::Mat normalized_disp_pred;
  disp.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                 -minVal * 255.0 / (maxVal - minVal));

  cv::Mat color_normalized_disp_pred;
  cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
  cv::imwrite("/workspace/test_data/lightstereo_result_color.png", color_normalized_disp_pred);
}


TEST(lightstereo_test, rknn_core_pipeline_speed)
{
  auto model         = CreateLightStereoModel();
  model->InitPipeline();
  auto [left, right] = ReadTestImages();

  deploy_core::BlockQueue<std::shared_ptr<std::future<cv::Mat>>> future_bq(100);

  auto func_push_data = [&]() {
    int index = 0;
    while (index++ < 2000)
    {
      auto p_fut = std::make_shared<std::future<cv::Mat>>(
          model->ComputeDispAsync(left.clone(), right.clone()));
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
      if (index ++ % 100 == 0) {
        LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
      }
    }
  };

  std::thread t_push(func_push_data);
  std::thread t_take(func_take_results);

  t_push.join();
  model->StopPipeline();
  t_take.join();
  model->ClosePipeline();

  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
}
