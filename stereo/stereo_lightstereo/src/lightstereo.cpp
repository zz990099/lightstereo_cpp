#include "stereo_lightstereo/lightstereo.hpp"

namespace stereo {

class LightStereo : public BaseStereoMatchingModel {
public:
  LightStereo(const std::shared_ptr<inference_core::BaseInferCore>      &infer_core,
              const std::shared_ptr<detection_2d::IDetectionPreProcess> &preprocess_block,
              const int                                                  input_height,
              const int                                                  input_width,
              const std::vector<std::string>                            &input_blobs_name,
              const std::vector<std::string>                            &output_blobs_name);

  ~LightStereo() = default;

private:
  bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) override;

  bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) override;

private:
  const std::vector<std::string> input_blobs_name_;
  const std::vector<std::string> output_blobs_name_;
  const int                      input_height_;
  const int                      input_width_;

  const std::shared_ptr<inference_core::BaseInferCore> infer_core_;
  std::shared_ptr<detection_2d::IDetectionPreProcess>  preprocess_block_;
};

LightStereo::LightStereo(
    const std::shared_ptr<inference_core::BaseInferCore>      &infer_core,
    const std::shared_ptr<detection_2d::IDetectionPreProcess> &preprocess_block,
    const int                                                  input_height,
    const int                                                  input_width,
    const std::vector<std::string>                            &input_blobs_name,
    const std::vector<std::string>                            &output_blobs_name)
    : BaseStereoMatchingModel(infer_core),
      infer_core_(infer_core),
      preprocess_block_(preprocess_block),
      input_height_(input_height),
      input_width_(input_width),
      input_blobs_name_(input_blobs_name),
      output_blobs_name_(output_blobs_name)
{
  // Check if the input arguments and inference_core matches
  auto p_map_buffer2ptr = infer_core_->AllocBlobsBuffer();
  if (p_map_buffer2ptr->Size() != input_blobs_name_.size() + output_blobs_name_.size())
  {
    LOG(ERROR) << "[LightStereo] Infer core should has {"
               << input_blobs_name_.size() + output_blobs_name_.size() << "} blobs !"
               << " but got " << p_map_buffer2ptr->Size() << " blobs";
    throw std::runtime_error("[LightStereo] Got invalid input arguments!!");
  }

  for (const std::string &input_blob_name : input_blobs_name)
  {
    if (p_map_buffer2ptr->GetOuterBlobBuffer(input_blob_name).first == nullptr)
    {
      LOG(ERROR) << "[LightStereo] Input_blob_name_ {" << input_blob_name
                 << "input blob name does not match `infer_core_` !";
      throw std::runtime_error("[LightStereo] Got invalid input arguments!!");
    }
  }

  for (const std::string &output_blob_name : output_blobs_name)
  {
    if (p_map_buffer2ptr->GetOuterBlobBuffer(output_blob_name).first == nullptr)
    {
      LOG(ERROR) << "[LightStereo] Output_blob_name_ {" << output_blob_name
                 << "} does not match name in infer_core_ !";
      throw std::runtime_error("[LightStereo] Got invalid input arguments!!");
    }
  }
}

bool LightStereo::PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<StereoPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[LightStereo] PreProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  const auto &p_blob_buffers = package->GetInferBuffer();
  const float left_scale     = preprocess_block_->Preprocess(
      package->left_image_data, p_blob_buffers, input_blobs_name_[0], input_height_, input_width_);
  const float right_scale = preprocess_block_->Preprocess(
      package->right_image_data, p_blob_buffers, input_blobs_name_[1], input_height_, input_width_);

  package->transform_scale = left_scale;
  return true;
}

bool LightStereo::PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<StereoPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[LightStereo] PostProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  auto p_blob_buffers = package->GetInferBuffer();

  const void *output_disp = p_blob_buffers->GetOuterBlobBuffer(output_blobs_name_[0]).first;
  CHECK_STATE(output_disp != nullptr,
              "[LightStereo] `PostProcess` Got invalid output disp ptr !!!");

  cv::Mat disp(input_height_, input_width_, CV_32FC1);
  memcpy(disp.data, output_disp, input_height_ * input_width_ * sizeof(float));

  // 1. crop
  const int original_height = package->left_image_data->GetImageDataInfo().image_height;
  const int original_width = package->left_image_data->GetImageDataInfo().image_width;
  const int crop_height = original_height * package->transform_scale;
  const int crop_width = original_width * package->transform_scale;
  cv::Mat crop_disp = disp(cv::Rect(0, 0, crop_width, crop_height));

  // 2. resize to original
  cv::Mat disp_to_original;
  cv::resize(crop_disp, disp_to_original, {original_width, original_height});

  package->disp = disp_to_original;

  
  return true;
}

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoModel(
    const std::shared_ptr<inference_core::BaseInferCore>      &infer_core,
    const std::shared_ptr<detection_2d::IDetectionPreProcess> &preprocess_block,
    const int                                                  input_height,
    const int                                                  input_width,
    const std::vector<std::string>                            &input_blobs_name,
    const std::vector<std::string>                            &output_blobs_name)
{
  return std::make_shared<LightStereo>(infer_core, preprocess_block, input_height, input_width,
                                       input_blobs_name, output_blobs_name);
}

} // namespace stereo