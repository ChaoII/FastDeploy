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
#include "fastdeploy_capi/core/fd_common.h"

#ifdef __cplusplus
extern "C" {
#endif
FASTDEPLOY_CAPI_EXPORT void FD_C_SetLogger(FD_C_Bool enable_info,
                                           FD_C_Bool enable_warning);

#ifdef __cplusplus
}
#endif