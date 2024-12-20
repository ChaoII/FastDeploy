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

#ifndef NON_64_PLATFORM

#include "adaptive_pool2d.h"

namespace fastdeploy {
void AdaptivePool2d::Compute(const Ort::Custom::Tensor<float>& input_tensor,
                             Ort::Custom::Tensor<float>& output_tensor) {
  std::cout << "--------AdaptivePool2d--------------" << std::endl;

  auto input_size = input_tensor.Shape();
  auto input_data = input_tensor.Data();
  auto output_data = output_tensor.Allocate(output_size_);

  int64_t in_bc_offset = input_size[2] * input_size[3];
  int64_t out_bc_offset = output_size_[2] * output_size_[3];
  for (int64_t b = 0; b < output_size_[0]; b++) {
    for (int64_t c = 0; c < output_size_[1]; c++) {
      for (int64_t h = 0; h < output_size_[2]; h++) {
        int64_t hstart =
            std::floor(static_cast<float>(h * input_size[2]) / output_size_[2]);
        int64_t hend = std::ceil(static_cast<float>((h + 1) * input_size[2]) /
                                 output_size_[2]);
        for (int64_t w = 0; w < output_size_[3]; w++) {
          int64_t wstart = std::floor(static_cast<float>(w * input_size[3]) /
                                      output_size_[3]);
          int64_t wend = std::ceil(static_cast<float>((w + 1) * input_size[3]) /
                                   output_size_[3]);
          int64_t out_offset = h * output_size_[3] + w;
          output_data[out_offset] = 0;
          for (auto i = hstart; i < hend; i++) {
            for (auto j = wstart; j < wend; j++) {
              if (pooling_type_ == "avg") {
                output_data[out_offset] += input_data[i * input_size[3] + j];
              }
              if (pooling_type_ == "max") {
                output_data[out_offset] = std::max(
                    output_data[out_offset], input_data[i * input_size[3] + j]);
              }
            }
          }
          if (pooling_type_ == "avg") {
            output_data[out_offset] /= ((hend - hstart) * (wend - wstart));
          }
        }
      }
      output_data += out_bc_offset;
      input_data += in_bc_offset;
    }
  }
}

}  // namespace fastdeploy

#endif
