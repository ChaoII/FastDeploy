// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "fastdeploy/runtime/backends/backend.h"
#include "fastdeploy/runtime/backends/mnn/option.h"
#include "fastdeploy/utils/unique_ptr.h"
#include <MNN/Interpreter.hpp>  // NOLINT
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fastdeploy {
class MNNBackend : public BaseBackend {
 public:
  /***************************** BaseBackend API *****************************/
  MNNBackend() = default;
  ~MNNBackend() override;
  bool Init(const RuntimeOption& runtime_option) override;
  int NumInputs() const override {
    return static_cast<int>(inputs_desc_.size());
  }
  int NumOutputs() const override {
    return static_cast<int>(outputs_desc_.size());
  }
  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;
  std::vector<TensorInfo> GetInputInfos() override;
  std::vector<TensorInfo> GetOutputInfos() override;
  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs,
             bool copy_to_fd) override;
  /***************************** BaseBackend API *****************************/

 private:
  void BuildOption(const RuntimeOption& option);
  FDDataType MNNTensorTypeToFDDataType(halide_type_t type);
  halide_type_t FDDataTypeToMNNTensorType(fastdeploy::FDDataType type);
  template <class SrcType, class DstType>
  std::vector<DstType> ConvertShape(std::vector<SrcType> shape) {
    std::vector<DstType> out_shape(shape.size());
    std::transform(
        shape.begin(), shape.end(), out_shape.begin(),
        [](const SrcType& value) { return static_cast<DstType>(value); });
    return out_shape;
  }

 private:
  MNNBackendOption option_;
  MNN::Session* session_ = nullptr;
  std::shared_ptr<MNN::Interpreter> net_;
  std::vector<TensorInfo> inputs_desc_;
  std::vector<TensorInfo> outputs_desc_;
};
}  // namespace fastdeploy
