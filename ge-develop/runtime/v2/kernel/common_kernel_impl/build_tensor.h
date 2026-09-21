/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_BUILD_TENSOR_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_BUILD_TENSOR_H_
#include "graph/types.h"
#include "register/kernel_registry.h"

namespace gert {
namespace kernel {
enum class BuildTensorInputs {
  kShape,
  kTensorData,
  kTensorAttr,
  kEnd
};
constexpr const ge::char_t *kBuildTensor = "BuildTensor";
ge::graphStatus BuildTensor(KernelContext *context);

ge::graphStatus BuildRefTensor(KernelContext *context);

constexpr const ge::char_t *kSplitTensorForOutputData = "SplitTensorForOutputData";
ge::graphStatus SplitTensorForOutputData(KernelContext *context);

constexpr const ge::char_t *kSplitDataTensor = "SplitDataTensor";
constexpr const ge::char_t *kSplitConstTensor = "SplitConstTensor";
enum class SplitTensorInputs{
  kTensor,
  kStreamId,
  kSize,
  kEnd
};
enum class SplitTensorForOutputDataInputs{
  kTensor,
  kAllocator,
  kSize,
  kEnd
};
enum class SplitTensorOutputs {
  kShape,
  kTensorData,
  kNum
};

constexpr const ge::char_t *kBuildShapeTensorData = "BuildShapeTensorData";

enum class BuildShapeTensorDataInputs {
  kNum
};
}  // namespace kernel
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_BUILD_TENSOR_H_
