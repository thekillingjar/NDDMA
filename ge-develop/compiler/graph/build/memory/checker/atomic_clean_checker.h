/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_CHECKER_ATOMIC_CLEAN_CHECKER_H
#define GE_GRAPH_BUILD_MEMORY_CHECKER_ATOMIC_CLEAN_CHECKER_H
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "ge/ge_api_types.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/build/memory/graph_mem_assigner.h"

namespace ge {
struct CleanAddrSizeType {
  bool operator<(const CleanAddrSizeType *other) const {
    return mem_info < other->mem_info;
  }
  OpMemoryType iow_type;
  int64_t index;
  CleanMemInfo mem_info;
  std::string ToString() const {
    std::stringstream ss;
    ss << NodeTypeIndex::GetMemType(iow_type) << "[" << index << "], " << mem_info.ToStr();
    return ss.str();
  }
};

class AtomicCleanChecker {
 public:
  explicit AtomicCleanChecker(const GraphMemoryAssigner *const assigner) : graph_memory_assigner_(assigner) {}
  // 内存分配完成后调用，用于检查需要进行集中清零的算子，其内存是不是在MemSet的workspace内存中
  Status Check(const ComputeGraphPtr &graph);
 private:
  Status GetMemSetAddrs(const Node *const node);
  Status GetCleanAddrs(const Node *const node, std::vector<CleanAddrSizeType> &clean_addrs) const;
  Status GetInputCleanAddrs(const Node *const node,
                            std::vector<CleanAddrSizeType> &clean_addrs) const;
  Status GetOutputCleanAddrs(const Node *const node, AtomicNodeCleanTypeVals &type_vals,
                             std::vector<CleanAddrSizeType> &clean_addrs) const;
  bool FindInMemSetOffsets(const std::vector<Node *> &memset_nodes,
                           const CleanAddrSizeType &addr_size_type,
                           const bool error_mode = false) const;
  Status GetMemType(const Node *const node, const IOType &io_type, const uint32_t index, uint32_t &mem_type) const;
  void PrintMemSet(const std::vector<Node *> &memset_nodes) const;
 private:
  const GraphMemoryAssigner *const graph_memory_assigner_;
  std::map<const Node *const, std::set<CleanMemInfo>> memset_to_clean_infos_;
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_CHECKER_ATOMIC_CLEAN_CHECKER_H
