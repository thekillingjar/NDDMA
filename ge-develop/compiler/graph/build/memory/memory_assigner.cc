/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/memory/memory_assigner.h"
#include <memory>
#include "framework/common/debug/ge_log.h"
#include "common/util/mem_utils.h"
#include "checker.h"

namespace ge {
Status MemoryAssigner::AssignMemory(std::map<uint64_t, size_t> &mem_offsets,
                                    size_t &zero_copy_mem_size, const bool has_assigned_var_mem) {
  graph_mem_assigner_ = MakeShared<GraphMemoryAssigner>(compute_graph_);
  GE_CHECK_NOTNULL(graph_mem_assigner_);
  if (graph_mem_assigner_->AssignMemory(has_assigned_var_mem) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Assign][Memory] failed, graph:%s", compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  // Reassign memory for special nodes
  Status ret_reassign = graph_mem_assigner_->ReAssignMemory(mem_offsets);
  if (ret_reassign != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[ReAssign][Memory] failed, graph:%s", compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  // Assign memory (block and offset) for zero copy nodes
  if (graph_mem_assigner_->AssignZeroCopyMemory(mem_offsets, zero_copy_mem_size) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Assign][ZeroCopyMemory] failed, graph:%s", compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  const auto graph_mem_splitter = graph_mem_assigner_->GetGraphMemSplitter();
  if (graph_mem_splitter != nullptr) {
    sub_mem_offsets_ = std::move(graph_mem_splitter->GetSubMemOffsets());
  }

  if (graph_mem_assigner_->AssignMemory2HasRefAttrNode() != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Assign][Memory] to node which has ref attr failed! graph:%s",
           compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  // Assign memory for reference
  if (graph_mem_assigner_->AssignReferenceMemory() != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Assign][ReferenceMemory] failed! graph:%s", compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  if (graph_mem_assigner_->ReAssignContinuousMemory() != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Assign][ReAssignContinuousMemory] failed, graph:%s", compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  // Must do variable attr assign after all the memory assigned
  if (graph_mem_assigner_->AssignVarAttr2Nodes() != SUCCESS) {
    GELOGE(FAILED, "[Variable][Memory] assigner failed, graph:%s", compute_graph_->GetName().c_str());
    return FAILED;
  }
  if (graph_mem_assigner_->SetInputOffset() != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Set][InputOffset] Fail! graph:%s", compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  GE_ASSERT_SUCCESS(graph_mem_assigner_->SetAtomicCleanOffset());

  if (graph_mem_assigner_->CheckOffset() != SUCCESS) {
    GELOGE(FAILED, "[Check][Offset] Fail! graph:%s", compute_graph_->GetName().c_str());
    return FAILED;
  }

  graph_mem_assigner_->MarkDistanceAttr();
  return SUCCESS;
}
}  // namespace ge
