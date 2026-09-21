/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_39BC87BB1C574BBB96A3716A54964F5F_H
#define INC_39BC87BB1C574BBB96A3716A54964F5F_H

#include <functional>
#include "ge/ge_api_types.h"
#include "graph/compute_graph.h"

namespace gert {
class GraphUnfolder {
 public:
  static ge::Status UnfoldSubgraphs(const ge::ComputeGraphPtr &root_graph, ge::ComputeGraphPtr &merged_graph);

  static ge::Status UnfoldAllPartitioncallInPlace(const ge::ComputeGraphPtr &root_graph);

  static bool IsGraphNeedUnfold(const ge::ComputeGraphPtr &root_graph);
  static bool IsGraphDynamicCompiled(const ge::ComputeGraphPtr &graph);

  static bool IsDataNotNeedRefConst(const ge::NodePtr &node);
  static bool IsDataNotNeedRefConst(const ge::Node *node);

 private:
    static ge::Status DoUnlinkDataAnchors(const ge::OutDataAnchorPtr &out_data_anchor,
                                          const ge::InDataAnchorPtr &in_data_anchor);
    static ge::Status DoLinkDataAnchors(const ge::OutDataAnchorPtr &out_data_anchor,
                                        const ge::InDataAnchorPtr &in_data_anchor);
    static ge::Status MergeInputNodes(ge::ComputeGraph &compute_graph);
    static ge::Status CheckInputInData(const ge::NodePtr &node, std::set<ge::NodePtr> &root_nodes);
    static ge::Status MergeInputInData(const ge::NodePtr &node, const ge::NodePtr &wrapped_node,
                                       std::set<ge::NodePtr> &root_nodes);
    static ge::Status MergeNetOutputNode(ge::ComputeGraph &compute_graph);
    static ge::Status MergeNetOutputInData(const ge::NodePtr &parent_node, const ge::OpDescPtr &net_output_desc,
                                           const ge::InDataAnchorPtr &in_data_anchor);
    static ge::Status UnfoldSubgraph(const ge::ComputeGraphPtr &root_graph,
                                     const ge::ComputeGraphPtr &origin_sub_graph,
                                     ge::ComputeGraphPtr &merged_graph,
                                     const uint32_t depth = 0U);
    static ge::Status UnfoldPartitionedCallSubgraph(const ge::ComputeGraphPtr &sub_graph,
                                                    ge::ComputeGraphPtr &merged_graph,
                                                    const ge::ComputeGraphPtr &root_graph,
                                                    const ge::NodePtr &node,
                                                    const uint32_t depth);
    static ge::Status UnfoldControlNodeSubgraph(const std::vector<ge::ComputeGraphPtr> &subgraphs,
                                                const ge::ComputeGraphPtr &root_graph,
                                                const ge::NodePtr &node,
                                                const uint32_t depth);
    static ge::Status MarkGraphNodeIndex(const ge::ComputeGraphPtr &merged_graph);
    // 递归展平所有的partitioncall
    static ge::Status UnfoldPartitioncallInPlace(const ge::ComputeGraphPtr &root_graph,
      const ge::ComputeGraphPtr &graph, uint32_t depth);
    // 展平子图上的partitioncall
    static ge::Status UnfoldSubGraphPartitioncall(const ge::ComputeGraphPtr &root_graph,
      const ge::ComputeGraphPtr &graph);
};
}

#endif
