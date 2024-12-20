// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <iostream>
#include <map>

#ifndef NON_64_PLATFORM
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "onnxruntime_lite_custom_op.h"

namespace fastdeploy {

struct MultiClassNMS {
  MultiClassNMS(const OrtApi* ort_api, const OrtKernelInfo* info) {
    ort_api->KernelInfoGetAttribute_int64(info, "background_label",
                                          &background_label);
    ort_api->KernelInfoGetAttribute_int64(info, "keep_top_k", &keep_top_k);
    ort_api->KernelInfoGetAttribute_float(info, "nms_eta", &nms_eta);
    ort_api->KernelInfoGetAttribute_float(info, "nms_threshold",
                                          &nms_threshold);
    ort_api->KernelInfoGetAttribute_int64(info, "nms_top_k", &nms_top_k);
    ort_api->KernelInfoGetAttribute_int64(info, "normalized", &normalized);
    ort_api->KernelInfoGetAttribute_float(info, "score_threshold",
                                          &score_threshold);

    std::cout << "background_label" << background_label << std::endl;
  }

  void Compute(const Ort::Custom::Tensor<float>& boxes_tensor,
               const Ort::Custom::Tensor<float>& scores_tensor,
               Ort::Custom::Tensor<float>& out_box_tensor,
               Ort::Custom::Tensor<int32_t>& out_index_tensor,
               Ort::Custom::Tensor<int32_t>& out_num_rois_tensor);
  void FastNMS(const float* boxes, const float* scores, const int& num_boxes,
               std::vector<int>* keep_indices);
  int NMSForEachSample(const float* boxes, const float* scores, int num_boxes,
                       int num_classes,
                       std::map<int, std::vector<int>>* keep_indices);

  int64_t background_label = -1;
  int64_t keep_top_k = -1;
  float nms_eta{};
  float nms_threshold = 0.7;
  int64_t nms_top_k{};
  int64_t normalized{};
  float score_threshold{};
};

}  // namespace fastdeploy

#endif
