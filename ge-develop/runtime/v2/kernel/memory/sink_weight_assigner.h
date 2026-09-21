/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_ASSIGNER_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_ASSIGNER_KERNEL_H_
#include <cstddef>
#include "graph/types.h"
#include "graph/utils/math_util.h"
#include "graph/def_types.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
namespace memory {
class SinkWeightAssigner {
 public:
  SinkWeightAssigner(void *mem_base, const size_t &size) : base_addr_(mem_base), total_mem_size_(size) {};
  void *Assign(const size_t &size, const size_t &offset) {
    size_t end_offset = 0;
    if (ge::AddOverflow(size, offset, end_offset) ||  (end_offset > total_mem_size_)) {
      GELOGE(ge::GRAPH_FAILED, "Inpus invalid. size[%zu] + offset[%zu] > total_mem_size[%zu]",
          size, offset, total_mem_size_);
      return nullptr;
    }
    return ge::ValueToPtr(ge::PtrToValue(base_addr_) + offset);
  };

 private:
  void *base_addr_;
  size_t total_mem_size_;
};
}
namespace kernel {
enum class AssignWeightMemoryInputs {
  kOffsetTensor,
  kAssigner,
  kStreamId,
  kNum
};
enum class AssignWeightMemoryOutputs {
  kTensorData,
  kNum
};
enum class GetOrCreateWeightInputs {
  kRequiredSize,
  kAppointedWeight,
  kAllocator,
  kNum
};
enum class GetOrCreateWeightOutputs {
  kTensorData,
  kNum
};
enum class CreateMemAssignerInputs {
  kBaseMem,
  kNum
};
constexpr const ge::char_t *kAssignWeightMemory = "AssignWeightMemory";
} // namespace memory
} // namespace gert
#endif // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_ASSIGNER_KERNEL_H_
