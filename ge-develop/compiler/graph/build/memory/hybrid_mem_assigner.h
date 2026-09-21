/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_HYBRID_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_HYBRID_MEM_ASSIGNER_H_

#include <memory>
#include "graph/build/memory/mem_assigner.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/compute_graph.h"
#include "graph/ge_local_context.h"

#include "framework/common/types.h"
#include "framework/common/util.h"
#include "checker/reuse_checker.h"

namespace ge {
using BlockMemAssignerPtr = std::shared_ptr<BlockMemAssigner>;

class BlockMemAssigner;

class HybridMemAssigner : public MemAssigner {
 public:
  explicit HybridMemAssigner(ge::ComputeGraphPtr compute_graph);

  HybridMemAssigner(const HybridMemAssigner &) = delete;

  HybridMemAssigner &operator=(const HybridMemAssigner &) = delete;

  ~HybridMemAssigner() override = default;

  Status Assign() override;

  const std::map<uint64_t, size_t> &GetMemOffsets() const { return mem_offsets_; }

  const std::map<uint64_t, MemoryStat> &GetMemoryStat() const { return memory_stat_; }

  BlockMemAssignerPtr GetPriorityAssinger() const { return priority_assigner_; }

  const AnchorToSymbol &GetAnchorToSymbol() const { return mem_assist_info_.anchor_to_symbol; }

  const SymbolToAnchors &GetSymbolToAnchors() const { return mem_assist_info_.symbol_to_anchors; }

  ReuseChecker *GetReuseChecker() { return reuse_checker_.get(); }
  void ReuseCheckerDeInit();
 private:
  static Status AssignMemory(BlockMemAssigner *block_assigner, size_t &mem_size, const GEThreadLocalContext &context);
  Status ReuseCheckerInit();
  std::map<uint64_t, size_t> mem_offsets_;
  std::map<uint64_t, MemoryStat> memory_stat_;

  ge::ComputeGraphPtr compute_graph_;

  BlockMemAssignerPtr priority_assigner_;

  MemAssistInfo mem_assist_info_;
  std::unique_ptr<ReuseChecker> reuse_checker_ = nullptr;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_HYBRID_MEM_ASSIGNER_H_
