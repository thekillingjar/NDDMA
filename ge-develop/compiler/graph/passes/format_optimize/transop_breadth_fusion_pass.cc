/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/format_optimize/transop_breadth_fusion_pass.h"

#include <set>
#include <string>

#include "framework/common/types.h"
#include "graph/ge_context.h"
#include "framework/common/ge_types.h"
#include "common/op/transop_util.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
namespace {
const size_t kDataInputNum = 1U;
const char *const kPerm = "perm";
void GetTransposePerm(const NodePtr &node, std::vector<int64_t> &perm) {
  // The caller guarantees that the pointer is not null
  if (node->GetType() == TRANSPOSED) {
    (void) AttrUtils::GetListInt(node->GetOpDesc(), kPerm, perm);
  } else if (node->GetType() == TRANSPOSE) {
    auto dim_idx = static_cast<uint32_t>(node->GetOpDesc()->GetInputIndexByName(kPerm));
    const GeTensor *weight = OpDescUtils::GetInputConstData(OpDescUtils::CreateOperatorFromNode(node), dim_idx);
    if (weight != nullptr) {
      const auto tensor_desc = weight->GetTensorDesc();
      const auto shape_size = tensor_desc.GetShape().GetShapeSize();
      if (tensor_desc.GetDataType() == DT_INT32) {
        const int32_t *weight_data = reinterpret_cast<const int32_t *>(weight->GetData().data());
        for (int64_t i = 0L; i < shape_size; ++i) {
          perm.emplace_back(static_cast<int64_t>(weight_data[i]));
        }
      } else if (tensor_desc.GetDataType() == DT_INT64) {
        const int64_t *weight_data = reinterpret_cast<const int64_t *>(weight->GetData().data());
        for (int64_t i = 0L; i < shape_size; ++i) {
          perm.emplace_back(weight_data[i]);
        }
      }
    } else {
      GELOGW("Node:%s has no const input, no need to fusion", node->GetName().c_str());
      perm.emplace_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(node.get())));
    }
  } else {
    GELOGW("Only support getting perm for Transpose and TransposeD");
  }
  return;
}

void MoveInCtrlEdgeToPeerOut(const NodePtr &node) {
  for (const auto &peer_out : node->GetOutDataNodes()) {
    if (GraphUtils::MoveInCtrlEdges(node, peer_out) != SUCCESS) {
      continue;
    }
    GELOGD("Move in_ctrl_edge from %s to %s", node->GetName().c_str(), peer_out->GetName().c_str());
  }
}

Status CopyInDataEdgeToPeerOutAsCtrlEdge(const NodePtr &src_node) {
  const auto src_in_nodes = src_node->GetInDataNodes();
  if (src_in_nodes.empty()) {
    return GRAPH_SUCCESS;
  }
  if (src_node->GetOutDataNodesSize() == 0U) {
    return GRAPH_SUCCESS;
  }
  auto dst_node = src_node->GetOutDataNodes().at(0U);
  std::unordered_set<NodePtr> exist_in_ctrl_nodes_set;
  auto exist_in_ctrl_nodes = dst_node->GetInControlNodes();
  if (!exist_in_ctrl_nodes.empty()) {
    exist_in_ctrl_nodes_set.insert(exist_in_ctrl_nodes.begin(), exist_in_ctrl_nodes.end());
  }

  const auto dst_ctrl = dst_node->GetInControlAnchor();
  for (const auto &in_node : src_in_nodes) {
    if (exist_in_ctrl_nodes_set.count(in_node) > 0U) {
      continue;
    }

    const auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), dst_ctrl);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Add ControlEdge from %s to %s failed, when copy control dependencies from %s to %s",
                        in_node->GetName().c_str(), dst_node->GetName().c_str(), src_node->GetName().c_str(),
                        dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed, when copy control dependencies from %s to %s",
             in_node->GetName().c_str(), dst_node->GetName().c_str(), src_node->GetName().c_str(),
             dst_node->GetName().c_str());
      return ret;
    }
  }
  GELOGD("Move in_data_edge from %s to %s", src_node->GetName().c_str(), dst_node->GetName().c_str());
  return GRAPH_SUCCESS;
}

/// Transop has multi input,
/// among other inputs other than data, if there is at least one non const,
/// indicates that the transop cannot judge whether it is the same type of transop through the output shape
bool CanGetNodeIdByShapes(const NodePtr &node) {
  // The caller guarantees that the pointer is not null
  auto in_nodes = node->GetInDataNodes();
  if (in_nodes.size() > kDataInputNum) {
    for (size_t index = kDataInputNum; index < in_nodes.size(); ++index) {
      auto in_node = in_nodes.at(index);
      if ((in_node != nullptr) && (!PassUtils::IsConstant(in_node))) {
        GELOGD("Can not get node id by shapes for node:%s, as having non const input node:%s.",
               node->GetName().c_str(), in_node->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}
}
Status TransOpBreadthFusionPass::Run(ge::ComputeGraphPtr graph) {
  if (graph == nullptr) {
    return SUCCESS;
  }
  // breadth fusion pass requires new topologic
  Status ret_topo = graph->TopologicalSorting();
  if (ret_topo != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Topological sorting for graph:%s failed", graph->GetName().c_str());
    GELOGE(ret_topo, "[Call][TopologicalSorting] for graph:%s failed.", graph->GetName().c_str());
    return ret_topo;
  }

  for (auto const &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    auto ids_to_trans_nodes = GetOutputTransOpNodes(node);
    for (auto const &id_to_trans_nodes : ids_to_trans_nodes) {
      if (id_to_trans_nodes.second.size() > 1) {
        GELOGI(
            "Begin to breath fusion output trans-op-nodes for %s,"
            " trans id %s, trans-op count %zu.",
            node->GetName().c_str(), id_to_trans_nodes.first.c_str(), id_to_trans_nodes.second.size());
        graphStatus status = Fusion(id_to_trans_nodes.second, graph);
        if (status != GRAPH_SUCCESS) {
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

std::string TransOpBreadthFusionPass::GetNodeId(const int32_t anchor_index, const NodePtr &node) {
  std::stringstream id;
  bool trans_data_type = false;
  bool trans_format = false;
  bool trans_shape = false;
  vector<int64_t> perm;

  GE_IF_BOOL_EXEC((node == nullptr) || (node->GetOpDesc() == nullptr),
                  REPORT_INNER_ERR_MSG("E19999", "Param node or its op_desc is nullptr, check invalid");
                  GELOGE(FAILED, "[Check][Param] Param node or its op_desc is nullptr"); return "");

  std::set<std::string> trans_shapes = { RESHAPE, EXPANDDIMS, SQUEEZE, SQUEEZEV2, SQUEEZEV3, UNSQUEEZEV3};
  std::set<std::string> trans_shape_and_format_perm = { TRANSPOSE, TRANSPOSED };
  if (node->GetType() == CAST) {
    trans_data_type = true;
  } else if (node->GetType() == EXPANDDIMS) {
    trans_format = true;
    trans_shape = true;
  } else if (trans_shape_and_format_perm.count(node->GetType()) > 0) {
    trans_format = true;
    trans_shape = true;
    (void) GetTransposePerm(node, perm);
  } else if (node->GetType() == TRANSDATA) {
    trans_data_type = true;
    trans_format = true;
    trans_shape = true;
  } else if (trans_shapes.count(node->GetType()) > 0) {
    trans_shape = true;
  } else if (node->GetType() == REFORMAT) {
    trans_format = true;
  }

  id << node->GetType() << '-' << anchor_index;
  // temp solution, we should not care about which stream the trans op on
  std::string stream_label;
  if (AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_STREAM_LABEL, stream_label)) {
    GELOGD("Get stream label %s for node %s, add it to fusion id.", stream_label.c_str(), node->GetName().c_str());
    id << '-' << stream_label;
  }
  std::string memory_optimization_policy;
  ge::GetContext().GetOption(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy);
  for (const auto &in_ctrl_node : node->GetInControlNodes()) {
    //                    c
    // switch-->Identity ---> node
    // the control edge from a identity node can not be removed
    if (in_ctrl_node->GetType() == IDENTITY || memory_optimization_policy == kMemoryPriority) {
      id << "-control-in-" << in_ctrl_node->GetName();
    }
  }
  // [Cascade pointer]
  const auto &input_desc = node->GetOpDesc()->MutableInputDesc(0);
  const auto &output_desc = node->GetOpDesc()->MutableOutputDesc(0);
  GE_CHECK_NOTNULL_EXEC(input_desc, return "");
  GE_CHECK_NOTNULL_EXEC(output_desc, return "");
  if (trans_data_type) {
    id << '-';
    id << static_cast<int32_t>(input_desc->GetDataType());
    id << '-';
    id << static_cast<int32_t>(output_desc->GetDataType());
  }
  if (trans_format) {
    id << '-';
    id << static_cast<int32_t>(input_desc->GetFormat());
    id << '-';
    id << static_cast<int32_t>(output_desc->GetFormat());
  }
  if (trans_shape) {
    id << '-';
    id << JoinDims(",", input_desc->GetShape().GetDims());
    id << '-';
    id << JoinDims(",", output_desc->GetShape().GetDims());
  }
  if (!perm.empty()) {
    id << '-';
    id << JoinDims(",", perm);
  }

  return id.str();
}

/**
 * Get all transform operators in the output of node.
 * @param node
 * @return std::map
 *     key   - transform operator identifer
 *     value - transform operator set
 */
std::map<std::string, std::vector<NodePtr>> TransOpBreadthFusionPass::GetOutputTransOpNodes(const NodePtr &node) {
  auto result = std::map<std::string, std::vector<NodePtr>>();
  if (node == nullptr) {
    return result;
  }
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor == nullptr) {
        continue;
      }

      auto peer_node = peer_in_anchor->GetOwnerNode();
      if (peer_node == nullptr) {
        continue;
      }

      if (TransOpUtil::IsTransOp(peer_node) &&
          (peer_in_anchor->GetIdx() == TransOpUtil::GetTransOpDataIndex(peer_node)) &&
          CanGetNodeIdByShapes(peer_node)) {
        auto output_node_id = GetNodeId(out_anchor->GetIdx(), peer_node);
        result[output_node_id].push_back(peer_node);
      }
    }
  }
  return result;
}

/**
 * Reserving Transform operators which with smaller topo index,
 * other transform operators's output edges merge to the reserved transform operator.
 * Removed transform operators have no output edges.
 * @param trans_nodes
 * @param graph
 */
graphStatus TransOpBreadthFusionPass::Fusion(const std::vector<NodePtr> &trans_nodes, ComputeGraphPtr &graph) const {
  if (trans_nodes.empty()) {
    return GRAPH_FAILED;
  }

  size_t min_index = 0;
  GE_CHECK_NOTNULL(trans_nodes[0]);
  auto op_desc = trans_nodes[0]->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  int64_t min_id = op_desc->GetId();
  size_t vec_size = trans_nodes.size();
  for (size_t i = 1; i < vec_size; i++) {
    GE_CHECK_NOTNULL(trans_nodes[i]);
    op_desc = trans_nodes[i]->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->GetId() < min_id) {
      min_index = i;
      min_id = op_desc->GetId();
    }
  }

  NodePtr node_remain = trans_nodes[min_index];
  for (size_t i = 0; i < trans_nodes.size(); ++i) {
    if (min_index == i) {
      continue;
    }
    // Move in_ctrl_edge from remove_node to its peer out node
    // To avoid cycle if there are control relation between trans_nodes[i] and node_remain.
    MoveInCtrlEdgeToPeerOut(trans_nodes[i]);
    CopyInDataEdgeToPeerOutAsCtrlEdge(trans_nodes[i]);
    auto status = GraphUtils::CopyInCtrlEdges(trans_nodes[i], node_remain);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    status = NodeUtils::MoveOutputEdges(trans_nodes[i], node_remain);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    // remove useless trans_node
    status = GraphUtils::IsolateNode(trans_nodes[i], {});
    if (status != GRAPH_SUCCESS) {
      return status;
    }

    status = GraphUtils::RemoveNodeWithoutRelink(graph, trans_nodes[i]);
    if (status != GRAPH_SUCCESS) {
      return status;
    }
    GELOGD("[Breadth fusion] Remove node %s from graph", trans_nodes[i]->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

std::string TransOpBreadthFusionPass::JoinDims(const std::string &sp, const std::vector<int64_t> &dims) const {
  std::stringstream ss;
  bool first = true;
  for (int64_t dim : dims) {
    if (first) {
      first = false;
    } else {
      ss << sp;
    }
    ss << dim;
  }
  return ss.str();
}

REG_PASS_OPTION("TransOpBreadthFusionPass").LEVELS(OoLevel::kO3);
}  // namespace ge
