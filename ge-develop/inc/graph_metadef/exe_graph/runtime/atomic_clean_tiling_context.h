/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATOMICCLEANTILINGCONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATOMICCLEANTILINGCONTEXT_H_
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/continuous_vector.h"
namespace gert {
class AtomicCleanTilingContext : public TilingContext {
 public:
  /**
   * 获取workspace size的列表
   * @return workspace size列表
   */
  const ContinuousVector *GetCleanWorkspaceSizes() const {
    return GetInputPointer<ContinuousVector>(0);
  }

  /**
   * 通过节点的输出index，获取需要清理的输出内存的大小
   * @param index 节点输出index
   * @return 需要清理的输出内存的大小
   */
  uint64_t GetCleanOutputSize(size_t index) const {
    return GetInputValue<uint64_t>(index + 1U);
  }
};
static_assert(std::is_standard_layout<AtomicCleanTilingContext>::value,
              "The class AtomicCleanTilingContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_ATOMICCLEANTILINGCONTEXT_H_
