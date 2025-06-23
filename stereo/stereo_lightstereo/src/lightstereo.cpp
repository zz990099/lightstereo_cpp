#include "stereo_lightstereo/lightstereo.hpp"

namespace easy_deploy {

class LightStereo : public BaseStereoMatchingModel {
public:
  LightStereo(const std::shared_ptr<BaseInferCore>        &infer_core,
              const std::shared_ptr<IDetectionPreProcess> &preprocess_block,
              const int                                    input_height,
              const int                                    input_width,
              const std::vector<std::string>              &input_blobs_name,
              const std::vector<std::string>              &output_blobs_name);

  ~LightStereo() = default;

private:
  bool PreProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) override;

  bool PostProcess(std::shared_ptr<IPipelinePackage> pipeline_unit) override;

private:
  const std::vector<std::string> input_blobs_name_;
  const std::vector<std::string> output_blobs_name_;
  const int                      input_height_;
  const int                      input_width_;

  const std::shared_ptr<BaseInferCore>  infer_core_;
  std::shared_ptr<IDetectionPreProcess> preprocess_block_;
};

LightStereo::LightStereo(const std::shared_ptr<BaseInferCore>        &infer_core,
                         const std::shared_ptr<IDetectionPreProcess> &preprocess_block,
                         const int                                    input_height,
                         const int                                    input_width,
                         const std::vector<std::string>              &input_blobs_name,
                         const std::vector<std::string>              &output_blobs_name)
    : BaseStereoMatchingModel(infer_core),
      infer_core_(infer_core),
      preprocess_block_(preprocess_block),
      input_height_(input_height),
      input_width_(input_width),
      input_blobs_name_(input_blobs_name),
      output_blobs_name_(output_blobs_name)
{
  // Check if the input arguments and inference_core matches
  auto blobs_tensor = infer_core_->AllocBlobsBuffer();
  if (blobs_tensor->Size() != input_blobs_name_.size() + output_blobs_name_.size())
  {
    LOG_ERROR("[LightStereo] Infer core should has {%d} blobs, but got {%d} blobs!",
              input_blobs_name_.size() + output_blobs_name_.size(), blobs_tensor->Size());
    throw std::runtime_error("[LightStereo] Got invalid input arguments!!");
  }

  for (const std::string &input_blob_name : input_blobs_name)
  {
    blobs_tensor->GetTensor(input_blob_name);
  }

  for (const std::string &output_blob_name : output_blobs_name)
  {
    blobs_tensor->GetTensor(output_blob_name);
  }
}

bool LightStereo::PreProcess(std::shared_ptr<IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<StereoPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[LightStereo] PreProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  auto blobs_tensor = package->GetInferBuffer();

  const float left_scale = preprocess_block_->Preprocess(
      package->left_image_data, blobs_tensor->GetTensor(input_blobs_name_[0]), input_height_,
      input_width_);
  const float right_scale = preprocess_block_->Preprocess(
      package->right_image_data, blobs_tensor->GetTensor(input_blobs_name_[1]), input_height_,
      input_width_);

  package->transform_scale = left_scale;
  return true;
}

bool LightStereo::PostProcess(std::shared_ptr<IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<StereoPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[LightStereo] PostProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  auto p_blob_buffers = package->GetInferBuffer();

  const void *output_disp = p_blob_buffers->GetTensor(output_blobs_name_[0])->RawPtr();
  CHECK_STATE(output_disp != nullptr,
              "[LightStereo] `PostProcess` Got invalid output disp ptr !!!");

  cv::Mat disp(input_height_, input_width_, CV_32FC1);
  memcpy(disp.data, output_disp, input_height_ * input_width_ * sizeof(float));
  disp /= package->transform_scale;

  // 1. crop
  const int original_height = package->left_image_data->GetImageDataInfo().image_height;
  const int original_width  = package->left_image_data->GetImageDataInfo().image_width;
  const int crop_height     = original_height * package->transform_scale;
  const int crop_width      = original_width * package->transform_scale;
  cv::Mat   crop_disp       = disp(cv::Rect(0, 0, crop_width, crop_height));

  // 2. resize to original
  cv::Mat disp_to_original;
  cv::resize(crop_disp, disp_to_original, {original_width, original_height});

  package->disp = disp_to_original;

  return true;
}

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoModel(
    const std::shared_ptr<BaseInferCore>        &infer_core,
    const std::shared_ptr<IDetectionPreProcess> &preprocess_block,
    const int                                    input_height,
    const int                                    input_width,
    const std::vector<std::string>              &input_blobs_name,
    const std::vector<std::string>              &output_blobs_name)
{
  return std::make_shared<LightStereo>(infer_core, preprocess_block, input_height, input_width,
                                       input_blobs_name, output_blobs_name);
}

} // namespace easy_deploy