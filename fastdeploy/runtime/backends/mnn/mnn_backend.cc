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
#include "fastdeploy/runtime/backends/mnn/mnn_backend.h"
#include <MNN/Tensor.hpp>
namespace fastdeploy {

MNNBackend::~MNNBackend() = default;

bool MNNBackend::Init(const RuntimeOption& runtime_option) {

  if (runtime_option.model_from_memory_) {
    FDERROR << "OpenVINOBackend doesn't support load model from memory, please "
               "load model from disk."
            << std::endl;
    return false;
  }

  if (initialized_) {
    FDERROR << "MNNBackend is already initialized, cannot initialize again."
            << std::endl;
    return false;
  }

  if (runtime_option.model_format != ModelFormat::MNNFormat) {
    FDERROR << "MNNBackend only supports model format MNN, but now it's "
            << runtime_option.model_format << "." << std::endl;
    return false;
  }
  auto interpreter =
      MNN::Interpreter::createFromFile(runtime_option.model_file.c_str());
  if (!interpreter) {
    FDERROR << "load mnn model file error, ensure model file is correct."
            << std::endl;
    return false;
  }

  net_.reset(interpreter, MNN::Interpreter::destroy);
  net_->setSessionMode(MNN::Interpreter::Session_Backend_Auto);
  net_->setSessionHint(MNN::Interpreter::MAX_TUNING_NUMBER, 5);

  BuildOption(runtime_option);

  auto mnn_inputs = net_->getSessionInputAll(session_);
  auto mnn_outputs = net_->getSessionOutputAll(session_);
  for (auto& input : mnn_inputs) {
    TensorInfo info;
    info.name = input.first;
    info.shape = input.second->shape();
    info.dtype = MNNTensorTypeToFDDataType(input.second->getType());
    inputs_desc_.emplace_back(info);
    net_->resizeTensor(input.second, input.second->shape());
  }
  net_->resizeSession(session_);
  for (auto& output : mnn_outputs) {
    TensorInfo info;
    info.name = output.first;
    info.shape = output.second->shape();
    info.dtype = MNNTensorTypeToFDDataType(output.second->getType());
    outputs_desc_.emplace_back(info);
  }
  initialized_ = true;
  return true;
}

TensorInfo MNNBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs())
  return inputs_desc_[index];
}
//
std::vector<TensorInfo> MNNBackend::GetInputInfos() { return inputs_desc_; }

TensorInfo MNNBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs %d.", index,
           NumOutputs())
  return outputs_desc_[index];
}

std::vector<TensorInfo> MNNBackend::GetOutputInfos() { return outputs_desc_; }

bool MNNBackend::Infer(std::vector<FDTensor>& inputs,
                       std::vector<FDTensor>* outputs, bool copy_to_fd) {

  if (inputs.size() != inputs_desc_.size()) {
    FDERROR << "[MNNBackend] Size of the inputs(" << inputs.size()
            << ") should keep same with the inputs of this model("
            << inputs_desc_.size() << ")." << std::endl;
    return false;
  }
  for (auto& input : inputs) {
    auto tensor = net_->getSessionInput(session_, input.name.c_str());
    auto mnn_tensor = new MNN::Tensor(tensor, MNN::Tensor::CAFFE);
    if (input.dtype == FDDataType::FP32) {
      memcpy(mnn_tensor->host<float>(), input.Data(), input.Nbytes());
    } else if (input.dtype == FDDataType::FP64) {
      memcpy(mnn_tensor->host<double>(), input.Data(), input.Nbytes());
    } else if (input.dtype == FDDataType::INT8) {
      memcpy(mnn_tensor->host<int8_t>(), input.Data(), input.Nbytes());
    } else if (input.dtype == FDDataType::INT16) {
      memcpy(mnn_tensor->host<int16_t>(), input.Data(), input.Nbytes());
    } else if (input.dtype == FDDataType::INT32) {
      memcpy(mnn_tensor->host<int32_t>(), input.Data(), input.Nbytes());
    } else if (input.dtype == FDDataType::UINT8) {
      memcpy(mnn_tensor->host<uint8_t>(), input.Data(), input.Nbytes());
    } else if (input.dtype == FDDataType::BOOL) {
      memcpy(mnn_tensor->host<bool>(), input.Data(), input.Nbytes());
    } else {
      FDASSERT(false, "Unexpected data type of %d.", input.dtype);
    }
    tensor->copyFromHostTensor(mnn_tensor);
    MNN::Tensor::destroy(mnn_tensor);
  }
  net_->runSession(session_);
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto tensor =
        net_->getSessionOutput(session_, outputs_desc_[i].name.c_str());
    if (outputs_desc_[i].dtype !=
        MNNTensorTypeToFDDataType(tensor->getType())) {
      outputs_desc_[i].dtype = MNNTensorTypeToFDDataType(tensor->getType());
    }
    std::vector<int64_t> temp_shape(tensor->shape().size());
    for (size_t s = 0; s < temp_shape.size(); s++) {
      temp_shape[s] = tensor->shape()[s];
    }
    outputs->resize(outputs_desc_.size());
    (*outputs)[i].Resize(temp_shape, outputs_desc_[i].dtype,
                         outputs_desc_[i].name);
    // nchw data format
    auto mnn_tensor = new MNN::Tensor(tensor, MNN::Tensor::CAFFE);
    tensor->copyToHostTensor(mnn_tensor);
    memcpy((*outputs)[i].MutableData(), mnn_tensor->host<float>(),
           (*outputs)[i].Nbytes());
    MNN::Tensor::destroy(mnn_tensor);
  }
  return true;
}

void MNNBackend::BuildOption(const RuntimeOption& option) {
  option_ = option.mnn_option;
  if (!option_.cache_file_path.empty()) {
    net_->setCacheFile(option_.cache_file_path.c_str());
  }
  MNN::ScheduleConfig config;
  MNN::BackendConfig backend_config;
  if (option.device == Device::CPU) {
    config.type = static_cast<MNNForwardType>(MNNForwardType::MNN_FORWARD_CPU);
    if (option_.cpu_thread_num > 0) {
      config.numThread = option_.cpu_thread_num;
    }
  } else if (option.device == Device::GPU) {
    config.type = static_cast<MNNForwardType>(option_.forward_type);
    config.mode = option_.gpu_mode;
    backend_config.precision =
        static_cast<MNN::BackendConfig::PrecisionMode>(option_.precision);
  }
  if (option_.power_mode != mnn::PowerMode::MNN_Power_Normal) {
#if defined(__aarch64__)
    backend_config.power =
        static_cast<MNN::BackendConfig::PowerMode>(option_.power_mode);
#else
    FDERROR << "power mode of MNN_Power_High and MNN_Power_High only be "
               "supported for aarch64 cpu , switch to MNN_Power_Normal"
            << std::endl;
#endif
  }
  backend_config.memory =
      static_cast<MNN::BackendConfig::MemoryMode>(option_.memory_mode);
  config.backendConfig = &backend_config;
  session_ = net_->createSession(config);
}

FDDataType MNNBackend::MNNTensorTypeToFDDataType(halide_type_t type) {

  if (type == halide_type_of<float>()) {
    return FDDataType::FP32;
  }
  if (type == halide_type_of<double>()) {
    return FDDataType::FP64;
  }
  if (type == halide_type_of<int8_t>()) {
    return FDDataType::INT8;
  }
  if (type == halide_type_of<int16_t>()) {
    return FDDataType::INT16;
  }
  if (type == halide_type_of<int32_t>()) {
    return FDDataType::INT32;
  }
  if (type == halide_type_of<uint8_t>()) {
    return FDDataType::UINT8;
  }
  if (type == halide_type_of<bool>()) {
    return FDDataType::BOOL;
  }
  FDERROR << "FDDataType don't support this type" << std::endl;
  return FDDataType::UNKNOWN1;
}

halide_type_t
MNNBackend::FDDataTypeToMNNTensorType(fastdeploy::FDDataType type) {

  if (type == FDDataType::FP32) {
    return halide_type_of<float>();
  }
  if (type == FDDataType::FP64) {
    return halide_type_of<double>();
  }
  if (type == FDDataType::INT8) {
    return halide_type_of<int8_t>();
  }
  if (type == FDDataType::INT16) {
    return halide_type_of<int16_t>();
  }
  if (type == FDDataType::INT32) {
    return halide_type_of<int32_t>();
  }
  if (type == FDDataType::UINT8) {
    return halide_type_of<uint8_t>();
  }
  if (type == FDDataType::BOOL) {
    return halide_type_of<bool>();
  }
  FDERROR << "rknn_tensor_type don't support this type" << std::endl;
  return halide_type_of<float>();
}

}  // namespace fastdeploy
