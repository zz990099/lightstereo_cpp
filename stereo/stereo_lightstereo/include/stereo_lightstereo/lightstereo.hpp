#pragma once

#include "deploy_core/base_stereo.hpp"
#include "deploy_core/base_detection.hpp"
#include "image_processing_utils/image_processing_utils.hpp"

namespace easy_deploy {

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoModel(
    const std::shared_ptr<BaseInferCore>        &infer_core,
    const std::shared_ptr<IImageProcessing> &preprocess_block,
    const int                                    input_height,
    const int                                    input_width,
    const std::vector<std::string>              &input_blobs_name,
    const std::vector<std::string>              &output_blobs_name);

} // namespace easy_deploy
