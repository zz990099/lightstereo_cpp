#pragma once

#include "deploy_core/base_detection.h"
#include "deploy_core/base_infer_core.h"
#include "deploy_core/wrapper.h"

namespace stereo {

struct StereoPipelinePackage : public async_pipeline::IPipelinePackage {
  // the wrapped pipeline image data
  std::shared_ptr<async_pipeline::IPipelineImageData> left_image_data;
  std::shared_ptr<async_pipeline::IPipelineImageData> right_image_data;
  // confidence used in postprocess
  float conf_thresh;
  // record the transform factor during image preprocess
  float transform_scale;

  //
  cv::Mat disp;

  // maintain the blobs buffer instance
  std::shared_ptr<inference_core::IBlobsBuffer> infer_buffer;

  // override from `IPipelinePakcage`, to provide the blobs buffer to inference_core
  std::shared_ptr<inference_core::IBlobsBuffer> GetInferBuffer() override
  {
    if (infer_buffer == nullptr)
    {
      LOG(ERROR) << "[DetectionPipelinePackage] returned nullptr of infer_buffer!!!";
    }
    return infer_buffer;
  }
};

class BaseStereoMatchingModel {
protected:
  BaseStereoMatchingModel(const std::shared_ptr<inference_core::BaseInferCore> &inference_core)
      : inference_core_(inference_core)
  {}

public:
  bool ComputeDisp(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disp_output)
  {
    CHECK_STATE(!left_image.empty() && !right_image.empty(),
                "[BaseStereoMatchingModel] `ComputeDisp` Got invalid input images !!!");

    auto package              = std::make_shared<StereoPipelinePackage>();
    package->left_image_data  = std::make_shared<PipelineCvImageWrapper>(left_image);
    package->right_image_data = std::make_shared<PipelineCvImageWrapper>(right_image);
    package->infer_buffer     = inference_core_->GetBuffer(true);
    CHECK_STATE(
        package->infer_buffer != nullptr,
        "[BaseStereoMatchingModel] `ComputeDisp` Got invalid inference core buffer ptr !!!");

    MESSURE_DURATION_AND_CHECK_STATE(
        PreProcess(package),
        "[BaseStereoMatchingModel] `ComputeDisp` Failed execute PreProcess !!!");
    MESSURE_DURATION_AND_CHECK_STATE(
        inference_core_->SyncInfer(package->infer_buffer),
        "[BaseStereoMatchingModel] `ComputeDisp` Failed execute inference sync infer !!!");
    MESSURE_DURATION_AND_CHECK_STATE(
        PostProcess(package),
        "[BaseStereoMatchingModel] `ComputeDisp` Failed execute PostProcess !!!");

    disp_output = std::move(package->disp);

    return true;
  }

protected:
  virtual bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) = 0;

  virtual bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) = 0;

protected:
  std::shared_ptr<inference_core::BaseInferCore> inference_core_;
};

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoModel(
    const std::shared_ptr<inference_core::BaseInferCore>      &infer_core,
    const std::shared_ptr<detection_2d::IDetectionPreProcess> &preprocess_block,
    const int                                                  input_height,
    const int                                                  input_width,
    const std::vector<std::string>                            &input_blobs_name,
    const std::vector<std::string>                            &output_blobs_name);

} // namespace stereo
