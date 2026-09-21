/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_KNOWN_SUBGRAPH_DAVINCI_MODEL_KERNEL_H
#define AIR_CXX_RUNTIME_V2_KERNEL_KNOWN_SUBGRAPH_DAVINCI_MODEL_KERNEL_H

#include <cstdint>
#include <cstddef>
namespace gert {
namespace kernel {
constexpr size_t kMaxFileNameLen = 255U; // linux file max len
enum class MemoryBaseType {
  kMemoryBaseTypeWeight,
  kMemoryBaseTypeFileConstant,
  kMemoryBaseTypeEnd
};

struct MemoryBaseTypeOffset {
  MemoryBaseType base_type;
  int64_t offset;
  uint64_t size;
};

enum class InputsCommon {
  kDavinciModel,
  kInputsCommonEnd
};

enum class InputsSpecial {
  kDavinciModel,
  kStreamId,
  kInputsCommonEnd
};

enum class ModelExecute {
  kStream = static_cast<int32_t>(InputsCommon::kInputsCommonEnd),
  kInputNum,
  kOutputNum,
  kModelExecuteEnd
};

enum class UpdateWorkspaces {
  kWorkspacesNum = static_cast<int32_t>(InputsCommon::kInputsCommonEnd),
  kWorkspaceMemory,
  kUpdateWorkspacesEnd
};

enum class DavinciModelCreateInput {
  kModelAddr,
  kSessionId,
  kAssignMem,
  kStepId,
  kRootGraphId,
  kSpaceRegistry,
  kFileConstantWeightDir,
  kRtStreamReuse,
  kFixedMemAddr,
  kFixedMemSize,
  kFixedMemTensorFromInit,
  kP2pFixedMemAddr,
  kP2pFixedMemSize,
  kP2pFixedMemTensorFromInit,
  kFrozenInputIndicies,
  kFileConstantUserMem,
  kDavinciModelCreateInputEnd
};

struct FileConstantNameAndMem {
  char name[kMaxFileNameLen + 1U];
  const void *mem;
  size_t size;
};

bool IsNeedMallocFixedMemoryOnInitGraph(const void *fixed_feature_mem, const size_t fixed_size);
} // namespace kernel
} // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_KNOWN_SUBGRAPH_DAVINCI_MODEL_KERNEL_H
