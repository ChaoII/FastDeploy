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
#include <string>
#include <vector>

namespace fastdeploy {
namespace mnn {
typedef enum {
  MNN_FORWARD_CPU = 0,
  /// Firstly find the first available backends not equal to CPU If no other backends, use cpu
  MNN_FORWARD_AUTO = 4,
  /// Hand write metal
  MNN_FORWARD_METAL = 1,
  /// NVIDIA GPU API
  MNN_FORWARD_CUDA = 2,
  /// Android / Common Device GPU API
  MNN_FORWARD_OPENCL = 3,
  MNN_FORWARD_OPENGL = 6,
  MNN_FORWARD_VULKAN = 7,
  /// Android 8.1's NNAPI or CoreML for ios
  MNN_FORWARD_NN = 5,
} MNNForwardType;

typedef enum {
  /// Forbidden tuning, performance not good
  MNN_GPU_TUNING_NONE = 1 << 0,
  /// heavily tuning, usually not suggested
  MNN_GPU_TUNING_HEAVY = 1 << 1,
  /// widely tuning, performance good. Default
  MNN_GPU_TUNING_WIDE = 1 << 2,
  /// normal tuning, performance may be ok
  MNN_GPU_TUNING_NORMAL = 1 << 3,
  /// ast tuning, performance may not good
  MNN_GPU_TUNING_FAST = 1 << 4,

  ///  User assign mode
  MNN_GPU_MEMORY_BUFFER = 1 << 6,
  ///  User assign mode
  MNN_GPU_MEMORY_IMAGE = 1 << 7,

  /// the kernels in one op execution record into one recording
  MNN_GPU_RECORD_OP = 1 << 8,
  /// 10 kernels record into one recording
  MNN_GPU_RECORD_BATCH = 1 << 9,
} MNNGpuMode;

typedef enum {
  MNN_Memory_Normal = 0,
  MNN_Memory_High,
  MNN_Memory_Low
} MemoryMode;

typedef enum { MNN_Power_Normal = 0, MNN_Power_High, MNN_Power_Low } PowerMode;

typedef enum {
  MNN_Precision_Normal = 0,
  MNN_Precision_High,
  MNN_Precision_Low,
  MNN_Precision_Low_BF16
} PrecisionMode;

}  // namespace mnn
struct MNNBackendOption {
  int device_id = 0;
  std::string cache_file_path;
  int cpu_thread_num = 4;
  mnn::MNNForwardType forward_type = mnn::MNN_FORWARD_AUTO;
  mnn::PrecisionMode precision = mnn::MNN_Precision_Normal;
  mnn::PowerMode power_mode = mnn::MNN_Power_Normal;
  mnn::MemoryMode memory_mode = mnn::MNN_Memory_Normal;
  int gpu_mode = mnn::MNN_GPU_TUNING_NONE;
};
}  // namespace fastdeploy
