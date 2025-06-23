#include "detection_2d_util/detection_2d_util.hpp"
#include "stereo_lightstereo/lightstereo.hpp"
#include "benchmark_utils/stereo_matching_benchmark_utils.hpp"

using namespace easy_deploy;

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.hpp"

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoTensorRTModel()
{
  auto engine = CreateTrtInferCore("/workspace/models/lightstereo_s_sceneflow_general_opt.engine");
  auto preprocess_block = CreateCudaDetPreProcess();
  return CreateLightStereoModel(engine, preprocess_block, 256, 512, {"left_img", "right_img"},
                                {"disp_pred"});
}

static void benchmark_stereo_matching_lightstereo_tensorrt_sync(benchmark::State &state)
{
  benchmark_stereo_matching_sync(state, CreateLightStereoTensorRTModel());
}
static void benchmark_stereo_matching_lightstereo_tensorrt_async(benchmark::State &state)
{
  benchmark_stereo_matching_async(state, CreateLightStereoTensorRTModel());
}
BENCHMARK(benchmark_stereo_matching_lightstereo_tensorrt_sync)->Arg(500)->UseRealTime();
BENCHMARK(benchmark_stereo_matching_lightstereo_tensorrt_async)->Arg(500)->UseRealTime();

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.hpp"

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoOnnxRuntimeModel()
{
  auto engine =
      CreateOrtInferCore("/workspace/models/lightstereo_s_sceneflow_general_opt.onnx",
                         {{"left_img", {1, 3, 256, 512}}, {"right_img", {1, 3, 256, 512}}},
                         {{"disp_pred", {1, 1, 256, 512}}});
  auto preprocess_block =
      CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);
  return CreateLightStereoModel(engine, preprocess_block, 256, 512, {"left_img", "right_img"},
                                {"disp_pred"});
}

static void benchmark_stereo_matching_lightstereo_onnxruntime_sync(benchmark::State &state)
{
  benchmark_stereo_matching_sync(state, CreateLightStereoOnnxRuntimeModel());
}
static void benchmark_stereo_matching_lightstereo_onnxruntime_async(benchmark::State &state)
{
  benchmark_stereo_matching_async(state, CreateLightStereoOnnxRuntimeModel());
}
BENCHMARK(benchmark_stereo_matching_lightstereo_onnxruntime_sync)->Arg(30)->UseRealTime();
BENCHMARK(benchmark_stereo_matching_lightstereo_onnxruntime_async)->Arg(30)->UseRealTime();

#endif

#ifdef ENABLE_RKNN

#include "rknn_core/rknn_core.hpp"

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoRknnModel()
{
  auto engine = CreateRknnInferCore(
      "/workspace/models/lightstereo_s_sceneflow_general_opt.rknn",
      {{"left_img", RknnInputTensorType::RK_UINT8}, {"right_img", RknnInputTensorType::RK_UINT8}},
      5, 3);
  auto preprocess_block = CreateCpuDetPreProcess({}, {}, false, false);
  return CreateLightStereoModel(engine, preprocess_block, 256, 512, {"left_img", "right_img"},
                                {"disp_pred"});
}

static void benchmark_stereo_matching_lightstereo_rknn_sync(benchmark::State &state)
{
  benchmark_stereo_matching_sync(state, CreateLightStereoRknnModel());
}
static void benchmark_stereo_matching_lightstereo_rknn_async(benchmark::State &state)
{
  benchmark_stereo_matching_async(state, CreateLightStereoRknnModel());
}
BENCHMARK(benchmark_stereo_matching_lightstereo_rknn_sync)->Arg(100)->UseRealTime();
BENCHMARK(benchmark_stereo_matching_lightstereo_rknn_async)->Arg(200)->UseRealTime();

#endif

BENCHMARK_MAIN();