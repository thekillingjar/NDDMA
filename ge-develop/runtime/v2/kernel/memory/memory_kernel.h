/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MEMORY_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MEMORY_KERNEL_H_
#include "graph/types.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/kernel_context.h"
namespace gert {
namespace kernel {
enum class AllocHbmMemInputs {
  kAllocator,
  kSizes,
  kEnds
};
constexpr ge::char_t const *kAllocHbmMemory = "AllocMemHbm";
constexpr ge::char_t const *kAllocMemory = "AllocMemory";
constexpr ge::char_t const *kAllocHostMemory = "AllocMemHost";
const std::string kFreeMemory = "FreeMemory";
const std::string kFreeMemoryHoldAddr = "FreeMemoryHoldAddr";
const std::string kFreeMemHbm = "FreeMemHbm";
const std::string kFreeMemHbmHoldAddr = "FreeMemHbmHoldAddr";
const std::string kFreeBatchHbm = "FreeBatchHbm";
const std::string kFreeBatchHbmHoldAddr = "FreeBatchHbmHoldAddr";

enum class MemoryType {
  kNodeOutput,
  kWorkspace,
  kShapeBuffer,
  kNum
};

enum class CreateAllocatorInputs {
  kPlacement,
  kMemoryType,
  kNum
};

enum class CreateL2AllocatorsInputs {
  kStreamNum,
  kPlacement,
  kNum
};

enum class CreateL2AllocatorsOutputs {
  kL2Allocators,
  kL2MemPools,
  kNum
};

enum class CreateInitL2AllocatorInputs {
  kL1Allocator,
  kL2Allocators,
  kStream,
  kNum
};

enum class CreateInitL2AllocatorOutputs {
  kInitL2Allocator,
  kL2AllocatorsForInitL2Allocator,
  kNum
};

enum class SelectL1AllocatorInputs {
  kPlacement,
  kExternalAllocator,
  kCreatedAllocator,
  kStream,
  kNum
};

enum class SelectL2AllocatorInputs {
  kStreamId,
  kStream,
  kL1Allocator,
  kL2Allocators,
  kNum
};

enum class GetAllocatorInputs {
  kPlacement,
  kExternalAllocator
};

enum class GerUserAllocatorOrFixedBaseAllocatorInputs {
  kPlacement,
  kSessionId,
  kStream
};

enum class AllocFixedFeatureMemoryInputs {
  kAllocator,
  kSize,
  kL2Allocator
};

enum class FreeFixedFeatureMemoryInputs {
  kGertTensorData
};
ge::graphStatus FreeHbmMemHoldAddr(KernelContext *context);
ge::graphStatus FreeBatchHbmHoldAddr(KernelContext *context);
ge::graphStatus FreeMemoryHoldAddr(KernelContext *context);
} // kernel
}
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MEMORY_KERNEL_H_
