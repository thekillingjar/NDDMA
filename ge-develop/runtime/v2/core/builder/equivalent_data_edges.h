/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_EQUIVALENT_DATA_EDGES_H_
#define AIR_CXX_RUNTIME_V2_CORE_EQUIVALENT_DATA_EDGES_H_

#include "graph/fast_graph/execute_graph.h"
#include "graph/ge_error_codes.h"

namespace gert {
struct EquivalentDataEdges {
  ge::graphStatus UpdateEquivalentEdges(ge::ExecuteGraph *const exe_graph);

 private:
  ge::graphStatus ConstructIoEquivalent(const std::vector<ge::FastNode *> &all_nodes);
  ge::graphStatus ConstructInnerDataOutputEquivalent(const std::vector<ge::FastNode *> &all_nodes);
  ge::graphStatus ConstructInnerDataEquivalent(const ge::FastNode *const node);
  ge::graphStatus ConstructInnerOutputEquivalent(const ge::FastNode *const node);
  ge::graphStatus ConstructRefOutputEquivalent(ge::ExecuteGraph *const exe_graph,
                                               const std::vector<ge::FastNode *> &all_nodes);
  ge::graphStatus SetInputOutputSymbols(const std::vector<ge::FastNode *> &all_nodes);

  // 0~31bit表示id, 32~62bit表示index, 63bit表示direction
  struct EndPoint {
    uint64_t id : 32;
    uint64_t index : 31;
    uint64_t direction : 1;

    std::string ToString() const {
      return std::string("EndPoint(")
          .append(std::to_string(id))
          .append(", ")
          .append(std::to_string(index))
          .append(", ")
          .append(std::to_string(direction))
          .append(")");
    }
  };
  uint64_t Encode(const EndPoint end_point) const;
  EndPoint Decode(uint64_t endpoint) const;

  void Add(uint64_t endpoint);
  void Merge(uint64_t endpoint1, uint64_t endpoint2);

  std::unordered_map<uint64_t, std::set<uint64_t>> symbols_to_endpoints_;
  std::unordered_map<uint64_t, uint64_t> endpoints_to_symbol_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_EQUIVALENT_DATA_EDGES_H_
