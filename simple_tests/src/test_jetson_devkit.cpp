#include <gtest/gtest.h>

#include "trt_core/trt_core.h"
#include "tests/fps_counter.h"
#include "detection_2d_util/detection_2d_util.h"
#include "stereo_lightstereo/lightstereo.hpp"

/**************************
****  trt core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;

TEST(simple_test, test)
{
  auto engine =
      CreateTrtInferCore("/workspace/models/lightstereo_s_sceneflow_general.engine");
  auto preprocess_block = CreateCudaDetPreProcess();
  auto model = stereo::CreateLightStereoModel(engine, preprocess_block, 256, 512, {"left_img", "right_img"}, {"disp_pred"});

  auto left = cv::imread("/workspace/test_data/left.png");
  auto right = cv::imread("/workspace/test_data/right.png");

  FPSCounter fps_counter;
  fps_counter.Start();
  for (int i = 0 ; i < 1000 ; ++ i) {
    cv::Mat disp;
    model->ComputeDisp(left, right, disp);
    fps_counter.Count(1);
    if (i % 100  == 0) {
      LOG(WARNING) << "cur fps : " << fps_counter.GetFPS();
      double minVal, maxVal;
      cv::minMaxLoc(disp, &minVal, &maxVal);
      cv::Mat normalized_disp_pred;
      disp.convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal),
                      -minVal * 255.0 / (maxVal - minVal));

      cv::Mat color_normalized_disp_pred;
      cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
      cv::imwrite("/workspace/test_data/lightstereo_result_color.png", color_normalized_disp_pred);
    }
  }
}
