#pragma once

#include "deploy_core/base_stereo.h"
#include "deploy_core/base_detection.h"

namespace stereo {

std::shared_ptr<BaseStereoMatchingModel> CreateLightStereoModel(
    const std::shared_ptr<inference_core::BaseInferCore>      &infer_core,
    const std::shared_ptr<detection_2d::IDetectionPreProcess> &preprocess_block,
    const int                                                  input_height,
    const int                                                  input_width,
    const std::vector<std::string>                            &input_blobs_name,
    const std::vector<std::string>                            &output_blobs_name);

} // namespace stereo
