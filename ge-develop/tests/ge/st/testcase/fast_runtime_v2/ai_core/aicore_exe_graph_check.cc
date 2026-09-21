/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore_exe_graph_check.h"
using namespace std;
using namespace ge;
namespace gert {
void GetAllLayerRecord(ExecuteGraph *exe_graph, std::vector<TestRecord> &record_v) {
  ExecuteGraph *tmp_graph = exe_graph;
  TestRecord record;
  while(tmp_graph->GetName().find("Main") == string::npos) {
    auto node11 = tmp_graph->GetParentNodeBarePtr();
    tmp_graph = node11->GetExtendInfo()->GetOwnerGraphBarePtr();
    if (tmp_graph == nullptr || (tmp_graph->GetName().find("ROOT") != string::npos)) {
      break;
    }
    record.node = node11;
    record.graph = tmp_graph;
    record_v.emplace_back(record);
  }
}

bool TravelFindEqualNode(const FastNode *node_pre, const FastNode *node_after) {
  GELOGD("Node[%s] dest name[%s].", node_pre->GetNamePtr(), node_after->GetNamePtr());
  if (node_pre == node_after) {
    return true;
  }
  for (auto node : node_pre->GetAllOutNodes()) {
    if (TravelFindEqualNode(node, node_after)) {
      return true;
    }
  }
  return false;
}

bool Judge2KernelPriority(ExecuteGraphPtr exe_graph, JudgeInfo &judge_info) {
  FastNode *node1 = nullptr;
  FastNode *node2 = nullptr;
  for (auto sub_graph : exe_graph->GetAllSubgraphs()) {
    GELOGD("Sub graph name[%s].", sub_graph->GetName().c_str());
    for (auto node : sub_graph->GetDirectNode()) {
      if ((node->GetType() == judge_info.high_pri_type) && (node->GetName().find(judge_info.high_pri_name) !=
                                                            string::npos)) {
        node1 = node;
      }
      if ((node->GetType() == judge_info.low_pri_type) && (node->GetName().find(judge_info.low_pri_name) !=
                                                           string::npos)) {
        node2 = node;
      }
    }
  }
  if (node1 == nullptr) {
    GELOGD("High priority node not found.");
    return false;
  }
  if (node2 == nullptr) {
    GELOGD("Low priority node not found.");
    return false;
  }
  ExecuteGraph *graph1 = node1->GetExtendInfo()->GetOwnerGraphBarePtr();
  ExecuteGraph *graph2 = node2->GetExtendInfo()->GetOwnerGraphBarePtr();
  TestRecord record;
  record.node = node1;
  record.graph = graph1;
  std::vector<TestRecord> record_v1 = {record};
  record.node = node2;
  record.graph = graph2;
  std::vector<TestRecord> record_v2 = {record};
  if (graph1->GetName() != graph2->GetName()) {
    GetAllLayerRecord(graph1, record_v1);
    GetAllLayerRecord(graph2, record_v2);
  }
  bool find = false;
  size_t i = 0;
  size_t j = 0;
  for (; i < record_v1.size(); ++i) {
    for (; j < record_v2.size(); ++j) {
      if (record_v1[i].graph == record_v2[j].graph) {
        find = true;
        break;
      }
    }
    if (find) {
      break;
    }
  }
  if (!find) {
    return false;
  }
  return TravelFindEqualNode(record_v1[i].node, record_v2[j].node);
}
}
