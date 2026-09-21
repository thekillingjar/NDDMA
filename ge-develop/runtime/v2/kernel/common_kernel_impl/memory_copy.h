/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_MEMORY_COPY_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_MEMORY_COPY_H_
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/common_kernel_impl/build_tensor.h"
namespace gert {
namespace kernel {
struct StorageTensorDesc {
  GertTensorData *tensor_data;
  size_t tensor_size;
  const StorageShape *storage_shape;
  ge::DataType data_type;
};
enum class MemoryCopyInputs {
  kSrcAddress,
  kSrcLength,
  kAllocator,
  kStream,
  kNum
};
enum class MemoryCopyOutputs {
  kAddress,
  kNum
};

enum class MakeSureTensorAtHostInputs {
  kStream,
  kAllocator,
  kAddrAndLengthStart,
};

enum class MakeSureTensorAtDeviceInputs {
  kStream,
  kAllocator,
  kAddrAndLengthStart
};

enum class MemoryCopyH2HInputs {
  kSrcAddress,
  kSrcShape,
  kSrcDataType,
  kAllocator,
  kNum
};

enum class MemoryCopyD2DInputs {
  kSrcAddress,
  kDstAddress,
  kTensorSize,
  kStream,
  kNum
};

enum class CalcStringTensorSizeInputs {
  kSrcDataType,
  kSrcShape,
  kSrcAddress,
  kStream
};

enum class EnsureTensorAtOutMemoryInputs {
  kStream = static_cast<size_t>(BuildTensorInputs::kEnd),
  kOutputData,
  kNum
};

enum class SinkWeightDataInputs {
  kWeightData,
  kDeviceBaseAddr,
  kStreamId,
  kStream,
  kNum
};

enum class SinkWeightDataOutputs {
  kTensorData,
  kNum
};
ge::graphStatus EnsureTensorAtOutMemory(KernelContext *context);
constexpr const char *kEnsureTensorAtOutMemory = "EnsureTensorAtOutMemory";
constexpr const ge::char_t *kMakeSureTensorAtDevice = "MakeSureTensorAtDevice";
constexpr const ge::char_t *kCopyH2D = "CopyH2D";
constexpr size_t kSizeOfCopyToDevice = 4U;
}
}
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_MEMORY_COPY_H_
