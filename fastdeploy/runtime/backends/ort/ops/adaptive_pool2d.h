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

#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/utils/utils.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <string>

#ifndef NON_64_PLATFORM
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "onnxruntime_lite_custom_op.h"

#ifdef WITH_GPU
#include "fastdeploy/runtime/backends/common/cuda/adaptive_pool2d_kernel.h"
#endif

namespace fastdeploy {

struct AdaptivePool2d {
  AdaptivePool2d(const OrtApi* ort_api, const OrtKernelInfo* info) {
    std::cout << "enter AdaptivePool2d" << std::endl;
    char* pooling_type = nullptr;
    size_t str_length = 0;
    ort_api->KernelInfoGetAttribute_string(info, "pooling_type", pooling_type,
                                           &str_length);
    int64_t* output_size = nullptr;
    size_t output_size_len = 0;
    pooling_type_ = std::string(pooling_type, str_length);
    ort_api->KernelInfoGetAttributeArray_int64(info, "output_size", output_size,
                                               &output_size_len);
    output_size_.assign(output_size, output_size + output_size_len);
  }

  std::string pooling_type_ = "avg";
  std::vector<int64_t> output_size_ = {};

  void Compute(const Ort::Custom::Tensor<float>& input_data,
               Ort::Custom::Tensor<float>& output_data);
};

}  // namespace fastdeploy

#endif
