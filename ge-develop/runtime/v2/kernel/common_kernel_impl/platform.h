/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_PLATFORM_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_PLATFORM_H_
#include "exe_graph/runtime/kernel_context.h"
#include "graph/ge_error_codes.h"

namespace gert {
namespace kernel {
enum class CoreTypeIndex {
  kAiCore,
  kVectorCore,
  kMix,
  kMixAic,
  kMixAiv,
  kMixAiCore,
  kMixAiVector,
  // add new core type here
  kOtherType
};

struct CoreNumInfos {
  int32_t soc_aicore_num;
  int32_t soc_vec_core_num;
  int32_t global_aicore_num;
  int32_t global_vec_core_num;
  int32_t op_aicore_num;
  int32_t op_vec_core_num;
};

ge::graphStatus GetPlatformInfo(KernelContext *context);
ge::graphStatus AppendCoreTypeToPlatform(KernelContext *context);
}  // namespace kernel
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_PLATFORM_H_
