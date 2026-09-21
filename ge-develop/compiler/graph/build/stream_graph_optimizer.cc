/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/stream_graph_optimizer.h"

#include <securec.h>

#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/tensor_utils.h"
#include "api/gelib/gelib.h"


namespace {
const int64_t kInvalidStream = -1;
}  // namespace
namespace ge {
StreamGraphOptimizer::~StreamGraphOptimizer() {}

void StreamGraphOptimizer::RefreshNodeId(const ComputeGraphPtr &comp_graph,
                                         Graph2SubGraphInfoList &subgraph_map) const {
  size_t node_size = comp_graph->GetAllNodesSize();
  GELOGD("Refresh placeholder and end nodeId start from node num: %zu", node_size);
  for (const auto &subgraph_pair : subgraph_map) {
    for (const auto &subgraph_info : subgraph_pair.second) {
      ComputeGraphPtr subgraph = subgraph_info->GetSubGraph();
      if (subgraph == nullptr) {
        continue;
      }
      for (ge::NodePtr &node : subgraph->GetDirectNode()) {
        GE_CHECK_NOTNULL_EXEC(node->GetOpDesc(), return);
        if ((node->GetType() == END) || (node->GetType() == PLACEHOLDER)) {
          node->GetOpDesc()->SetId(static_cast<int64_t>(node_size));
          node_size++;
        }
      }
    }
  }
}

bool StreamGraphOptimizer::IsSameStreamIdOrBatchLabel(const ComputeGraphPtr &comp_graph) const {
  if (comp_graph == nullptr) {
    return false;
  }
  std::set<int64_t> stream_set;
  std::set<std::string> label_set;
  for (const ge::NodePtr &cur_node : comp_graph->GetDirectNode()) {
    GE_IF_BOOL_EXEC(cur_node->GetOpDesc() == nullptr, continue);
    int64_t stream_id = cur_node->GetOpDesc()->GetStreamId();
    if (stream_id == kInvalidStream) {
      continue;
    }
    stream_set.insert(stream_id);

    std::string batch_label;
    if (AttrUtils::GetStr(cur_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label)) {
      label_set.insert(batch_label);
    } else {
      GELOGD("Node %s[%s] has no batch label, subgraph %s, stream id: %ld", cur_node->GetName().c_str(),
             cur_node->GetType().c_str(), comp_graph->GetName().c_str(), stream_id);
      continue;
    }

    GELOGD("Node %s in subgraph %s stream id: %ld, node num: %zu", cur_node->GetName().c_str(),
           comp_graph->GetName().c_str(), stream_id, comp_graph->GetDirectNodesSize());
  }
  if (stream_set.size() > 1 || label_set.size() > 1) {
    GELOGI("Nodes of graph: %s have different stream id or batch_label, node num: %zu, different stream num: %zu.",
           comp_graph->GetName().c_str(), comp_graph->GetDirectNodesSize(), stream_set.size());
    return false;
  }

  if (!label_set.empty()) {
    (void)AttrUtils::SetStr(comp_graph, ATTR_NAME_BATCH_LABEL, *label_set.begin());
  }
  return true;
}

Status StreamGraphOptimizer::OptimizeStreamedSubGraph(const ComputeGraphPtr &comp_graph,
                                                      Graph2SubGraphInfoList &subgraph_map,
                                                      struct RunContext &run_context) const {
  RefreshNodeId(comp_graph, subgraph_map);

  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  GE_CHECK_NOTNULL(instance);

  for (const auto &subgraph_pair : subgraph_map) {
    for (const auto &subgraph_info : subgraph_pair.second) {
      ComputeGraphPtr subgraph = subgraph_info->GetSubGraph();
      GE_CHECK_NOTNULL(subgraph);

      GELOGD("Optimize subgraph %s", subgraph->GetName().c_str());

      std::string engine_name = subgraph_info->GetEngineName();

      std::vector<GraphOptimizerPtr> graph_optimizers;
      if (instance->DNNEngineManagerObj().IsEngineRegistered(engine_name)) {
        instance->OpsKernelManagerObj().GetGraphOptimizerByEngine(engine_name, graph_optimizers);
        GELOGI("Subgraph: %s start optimize streamed graph. engineName: %s, graph Optimizer num: %zu.",
               subgraph->GetName().c_str(), engine_name.c_str(), graph_optimizers.size());

        auto nodes = subgraph->GetDirectNode();
        if (nodes.empty()) {
          continue;
        }

        if (!IsSameStreamIdOrBatchLabel(subgraph)) {
          GELOGI("There are more than one stream or batch_label in subgraph %s", subgraph->GetName().c_str());
          continue;
        }
        OpDescPtr op_desc = nodes.at(0)->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        int64_t stream_id = op_desc->GetStreamId();
        std::string batch_label;
        (void)AttrUtils::GetStr(subgraph, ATTR_NAME_BATCH_LABEL, batch_label);
        GELOGD("Subgraph has same stream id, subgraph: %s, engine_name: %s, stream_id: %ld, batch_label: %s",
               subgraph->GetName().c_str(), engine_name.c_str(), stream_id, batch_label.c_str());
        for (auto iter = graph_optimizers.begin(); iter != graph_optimizers.end(); ++iter) {
          GE_CHECK_NOTNULL(*iter);
          Status ret = (*iter)->OptimizeStreamGraph(*subgraph, run_context);
          if (ret != SUCCESS) {
            REPORT_INNER_ERR_MSG("E19999", "Call optimize streamed subgraph failed, subgraph: %s, engine_name: %s, graph "
                              "Optimizer num: %zu, ret: %u", subgraph->GetName().c_str(), engine_name.c_str(),
                              graph_optimizers.size(), ret);
            GELOGE(ret, "[Optimize][StreamGraph] failed, subgraph: %s, engine_name: %s, graph "
                   "Optimizer num: %zu, ret: %u",
                   subgraph->GetName().c_str(), engine_name.c_str(), graph_optimizers.size(), ret);
            return ret;
          }
          GELOGD("[optimizeStreamedSubGraph]: optimize streamed subgraph success, subgraph: %s, engine_name: %s, "
                 "graph Optimizer num: %zu!",
                 subgraph->GetName().c_str(), engine_name.c_str(), graph_optimizers.size());
        }
      }
    }
  }

  GELOGD("Optimize streamed subgraph success.");
  return SUCCESS;
}

Status StreamGraphOptimizer::OptimizeStreamedWholeGraph(const ComputeGraphPtr &comp_graph) const {
  GE_CHECK_NOTNULL(comp_graph);
  std::shared_ptr<GELib> instanece_ptr = GELib::GetInstance();
  if ((instanece_ptr == nullptr) || !instanece_ptr->InitFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Gelib is not inited before, check valid, graph:%s", comp_graph->GetName().c_str());
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Gelib is not inited before, graph:%s.",
           comp_graph->GetName().c_str());
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  GE_CHECK_NOTNULL(instanece_ptr->GetInstance());
  const auto &graph_optimizers =
      instanece_ptr->GetInstance()->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGI("optimize by opskernel in OptimizeStreamedWholeGraph. num of graph_optimizer is %zu.",
         graph_optimizers.size());

  for (const auto &iter : graph_optimizers) {
    GELOGI("Begin to optimize streamed whole graph by engine %s.", iter.first.c_str());
    GE_CHECK_NOTNULL(iter.second);
    Status ret = iter.second->OptimizeStreamedWholeGraph(*comp_graph);
    GE_ASSERT_SUCCESS(ret, "[Call][OptimizeStreamedWholeGraph] failed, ret:%u, engine_name:%s, graph_name:%s",
                      iter.first.c_str(), comp_graph->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge
