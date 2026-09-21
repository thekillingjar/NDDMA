/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/node_item.h"

#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "hybrid/executor/worker/shape_inference_engine.h"
#include "common/profiling/profiling_manager.h"
#include "hybrid/node_executor/node_executor.h"
#include "engines/local_engine/common/constant/constant.h"

namespace ge {
namespace hybrid {
namespace {
const uint8_t kMaxTransCount = 3U;
const uint8_t kTransOpIoSize = 1U;
const ge::char_t *const kAttrNameOriginalFusionGraph = "_original_fusion_graph";
const ge::char_t *const kNodeTypeRetVal = "_RetVal";
const std::set<std::string> kControlOpTypes {
    IF, STATELESSIF, CASE, STATELESSCASE, WHILE, STATELESSWHILE
};

const std::set<std::string> kControlFlowOpTypes {
    STREAMACTIVE, STREAMSWITCH, ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION, EXIT, REFEXIT,
    LABELGOTOEX, LABELSWITCHBYINDEX
};

const std::set<std::string> kMergeOpTypes {
    MERGE, REFMERGE, STREAMMERGE
};

bool IsEnterFeedNode(NodePtr node) {
  // For: Enter -> node
  // For: Enter -> Cast -> node
  // For: Enter -> TransData -> Cast -> node
  // For: Enter -> Const/Constant
  for (uint8_t i = 0U; i < kMaxTransCount; ++i) {
    if (kEnterOpTypes.count(NodeUtils::GetNodeType(node)) > 0U) {
      GELOGD("Node[%s] is Enter feed node.", node->GetName().c_str());
      return true;
    }
    if (kConstOpTypes.count(NodeUtils::GetNodeType(node)) != 0U) {
      auto in_ctrl_nodes = node->GetInControlNodes();
      auto nodes_size = in_ctrl_nodes.size();
      GELOGD("Node[%s] has in_ctrl_nodes size: %zu.", node->GetName().c_str(), nodes_size);
      if (nodes_size == kTransOpIoSize) {
        node = in_ctrl_nodes.at(0U);
        continue;
      }
    }
    if (!node->GetInControlNodes().empty()) {
      GELOGD("Node[%s] is controlled by other node except enter.", node->GetName().c_str());
      return false;
    }

    const auto all_data_nodes = node->GetInDataNodes();
    if ((all_data_nodes.size() != kTransOpIoSize) || (node->GetAllInDataAnchorsSize() != kTransOpIoSize)) {
      return false;
    }
    node = all_data_nodes.at(0U);
  }
  return false;
}

// Const (folding from enter)
//   |
// PartitionedCall (Transdata and so on)
//   |
// Mul (in loop)
// this scene should mark PartitionedCall as root node
bool CheckIsOutLoop(NodePtr node) {
  if (node->GetType() != PARTITIONEDCALL) {
    return false;
  }

  for (auto input_node : node->GetInAllNodes()) {
    if (!input_node->GetInAllNodes().empty()) {
      return false;
    }
  }
  return true;
}

Status ParseInputMapping(const Node &node, const OpDesc &op_desc, FusedSubgraph &fused_subgraph) {
  uint32_t parent_index = 0U;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(FAILED, "[Get][Attr][%s(%s)] Failed to get attr [%s]",
           op_desc.GetName().c_str(), op_desc.GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
    REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Failed to get attr [%s]",
                      op_desc.GetName().c_str(), op_desc.GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return FAILED;
  }

  for (auto &node_and_anchor : node.GetOutDataNodesAndAnchors()) {
    const auto &dst_op_desc = node_and_anchor.first->GetOpDesc();
    GE_CHECK_NOTNULL(dst_op_desc);
    const auto in_idx = node_and_anchor.second->GetIdx();
    auto tensor_desc = dst_op_desc->MutableInputDesc(static_cast<uint32_t>(in_idx));
    fused_subgraph.input_mapping[static_cast<int32_t>(parent_index)].emplace_back(tensor_desc);
    GELOGD("Input[%u] mapped to [%s:%u]", parent_index, dst_op_desc->GetName().c_str(), in_idx);
  }

  return SUCCESS;
}

Status ParseOutputMapping(const OpDescPtr &op_desc, FusedSubgraph &fused_subgraph) {
  uint32_t parent_index = 0U;
  for (const auto &input_desc : op_desc->GetAllInputsDescPtr()) {
    if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "[Get][Attr][%s(%s)] Failed to get attr [%s]", op_desc->GetName().c_str(),
             op_desc->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Failed to get attr [%s].", op_desc->GetName().c_str(),
                        op_desc->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return FAILED;
    }
    (void)fused_subgraph.output_mapping.emplace(static_cast<int32_t>(parent_index), input_desc);
  }
  return SUCCESS;
}

Status ReplaceRetValByNetOutput(ComputeGraphPtr &fused_graph) {
   // replace all retval to one netoutput, cause fused_graph may execute later.
  const auto netoutput_op_desc = MakeShared<OpDesc>(fused_graph->GetName() + "_netoutput", NETOUTPUT);
  GE_CHECK_NOTNULL(netoutput_op_desc);
  std::vector<NodePtr> ret_val_nodes;
  for (const auto &node : fused_graph->GetAllNodes()) {
    const std::string node_type = NodeUtils::GetNodeType(node);
    if (node_type == kNodeTypeRetVal) {
      ret_val_nodes.emplace_back(node);
    }
  }
  if (ret_val_nodes.empty()) {
    GELOGD("Fused graph %s has no ret val node.", fused_graph->GetName().c_str());
    return SUCCESS;
  }
  // cp ret_val input desc to netoutput
  for (const auto &node : ret_val_nodes) {
    const auto ret_val_op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(ret_val_op_desc);
    uint32_t parent_index = 0U;
    if (!AttrUtils::GetInt(ret_val_op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "[Get][Attr][%s(%s)] Failed to get attr [%s]", ret_val_op_desc->GetName().c_str(),
             ret_val_op_desc->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Failed to get attr [%s]", ret_val_op_desc->GetName().c_str(),
                        ret_val_op_desc->GetType().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return FAILED;
    }
    auto ret_val_input_desc = ret_val_op_desc->GetInputDesc(0U);
    (void)AttrUtils::SetInt(ret_val_input_desc, ATTR_NAME_PARENT_NODE_INDEX, static_cast<int64_t>(parent_index));
    (void)netoutput_op_desc->AddInputDesc(ret_val_input_desc);
  }

  const auto netoutput_node = fused_graph->AddNode(netoutput_op_desc);
  GE_CHECK_NOTNULL(netoutput_node);

  // move ret_val input edge to netoutput
  for (size_t i = 0U; i < ret_val_nodes.size(); ++i) {
    const auto in_node_and_anchors = ret_val_nodes[i]->GetInDataNodesAndAnchors();
    if (in_node_and_anchors.size() != 1U) {
      GELOGE(FAILED, "RetVal %s has more than one input.", ret_val_nodes[i]->GetName().c_str());
      return FAILED;
    }
    const auto ret_val_peer_out_anchor = in_node_and_anchors.at(0U).second;
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::ReplaceEdgeDst(ret_val_peer_out_anchor, ret_val_nodes[i]->GetInDataAnchor(0U),
                                                       netoutput_node->GetInDataAnchor(static_cast<int32_t>(i))));
  }
  // remove all retval
  for (const auto &node : ret_val_nodes) {
    GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(fused_graph, node));
  }
  GELOGD("Replace ret val node by netoutput.");
  return SUCCESS;
}

Status ParseFusedSubgraph(NodeItem &node_item) {
  if (!node_item.op_desc->HasAttr(kAttrNameOriginalFusionGraph)) {
    return SUCCESS;
  }

  GELOGI("[%s] Start to parse fused subgraph.", node_item.node_name.c_str());
  auto fused_subgraph = std::unique_ptr<FusedSubgraph>(new(std::nothrow)FusedSubgraph());
  GE_CHECK_NOTNULL(fused_subgraph);

  ComputeGraphPtr fused_graph;
  (void)AttrUtils::GetGraph(*node_item.op_desc, kAttrNameOriginalFusionGraph, fused_graph);
  GE_CHECK_NOTNULL(fused_graph);

  fused_graph->SetGraphUnknownFlag(true);
  fused_subgraph->graph = fused_graph;
  GE_CHK_GRAPH_STATUS_RET(fused_graph->TopologicalSorting());

  GE_CHK_GRAPH_STATUS_RET(ReplaceRetValByNetOutput(fused_graph));

  for (auto &node : fused_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const std::string node_type = NodeUtils::GetNodeType(node);
    if (node_type == DATA) {
      GE_CHK_GRAPH_STATUS_RET(ParseInputMapping(*node, *op_desc, *fused_subgraph));
      op_desc->SetOpEngineName(ge_local::kGeLocalEngineName);
      op_desc->SetOpKernelLibName(ge_local::kGeLocalOpKernelLibName);
    } else if (node_type == NETOUTPUT) {
      GE_CHK_GRAPH_STATUS_RET(ParseOutputMapping(op_desc, *fused_subgraph));
    } else {
      fused_subgraph->nodes.emplace_back(node);
    }
  }

  node_item.fused_subgraph = std::move(fused_subgraph);
  // register profiling name
  node_item.fused_subgraph->name_to_prof_indexes.resize(node_item.fused_subgraph->nodes.size());
  for (size_t i = 0UL; i < node_item.fused_subgraph->nodes.size(); ++i) {
    ProfilingManager::Instance().RegisterElement(node_item.fused_subgraph->name_to_prof_indexes[i],
                                                 node_item.fused_subgraph->nodes[i]->GetName());
  }

  GELOGI("[%s] Done parsing fused subgraph successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}
}  // namespace

bool IsControlFlowV2Op(const std::string &op_type) {
  return kControlOpTypes.count(op_type) > 0U;
}

bool IsFftsGraphNode(const OpDesc &op_desc) {
  return op_desc.HasAttr(ATTR_NAME_FFTS_SUB_GRAPH) || op_desc.HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH);
}

bool IsFftsKernelNode(const OpDesc &op_desc) {
  return op_desc.HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME);
}

NodeItem::NodeItem(NodePtr node_ptr) : NodeShapeInfer(), node(std::move(node_ptr)) {
  this->op_desc = this->node->GetOpDesc().get();
  this->node_name = this->node->GetName();
  this->node_type = this->node->GetType();
}

Status NodeItem::Create(const NodePtr &node_ptr, std::unique_ptr<NodeItem> &node_item) {
  GE_CHECK_NOTNULL(node_ptr);
  GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
  std::unique_ptr<NodeItem> instance(new(std::nothrow)NodeItem(node_ptr));
  GE_CHECK_NOTNULL(instance);
  GE_CHK_STATUS_RET(instance->Init(), "[Invoke][Init]Failed to init NodeItem [%s(%s)].",
                    node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
  node_item = std::move(instance);
  return SUCCESS;
}

void NodeItem::ResolveOptionalInputs() {
  if (op_desc->GetAllInputsSize() != op_desc->GetInputsSize()) {
    has_optional_inputs = true;
    for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
      const auto &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
      if (input_desc == nullptr) {
        GELOGD("[%s] Input[%zu] is optional and invalid", NodeName().c_str(), i);
      } else {
        input_desc_indices_.emplace_back(static_cast<uint32_t>(i));
      }
    }
  }
}

Status NodeItem::InitInputsAndOutputs() {
  GE_CHECK_LE(op_desc->GetInputsSize(), static_cast<uint32_t>(INT32_MAX));
  GE_CHECK_LE(op_desc->GetOutputsSize(), static_cast<uint32_t>(INT32_MAX));
  num_inputs = static_cast<int32_t>(op_desc->GetInputsSize());
  num_outputs = static_cast<int32_t>(op_desc->GetOutputsSize());
  if (AttrUtils::GetInt(op_desc, ::ge::ATTR_STAGE_LEVEL, group)) {
    GELOGD("[%s] Got stage level from op_desc = %d", op_desc->GetName().c_str(), group);
  } else {
    if (node->GetOwnerComputeGraph() != nullptr) {
      if (AttrUtils::GetInt(node->GetOwnerComputeGraph(), ::ge::ATTR_STAGE_LEVEL, group)) {
        GELOGD("[%s] Got stage level from parent graph = %d", op_desc->GetName().c_str(), group);
      } else {
        const auto &parent_node = node->GetOwnerComputeGraph()->GetParentNode();
        if ((parent_node != nullptr) && (AttrUtils::GetInt(parent_node->GetOpDesc(), ::ge::ATTR_STAGE_LEVEL, group))) {
          GELOGD("[%s] Got stage level from parent node = %d", op_desc->GetName().c_str(), group);
        } else {
          GELOGD("[%s] Node do not set stage level", op_desc->GetName().c_str());
        }
      }
    }
  }
  // set output mem type
  GE_CHK_STATUS_RET(ResolveOutputMemType(), "[Resolve][OutputMemType][%s(%s)] failed .", node->GetName().c_str(),
                    node->GetType().c_str());

  ResolveOptionalInputs();
  return SUCCESS;
}

Status NodeItem::ResolveOutputMemType() {
  const auto outputs_size = op_desc->GetOutputsSize();
  output_mem_types_.resize(outputs_size, MemStorageType::HBM);
  std::vector<int64_t> v_memory_type;
  const bool has_mem_type_attr = AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != outputs_size)) {
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size, op_desc->GetName().c_str(),
           op_desc->GetType().c_str());
    return FAILED;
  }
  for (size_t i = 0UL; i < outputs_size; ++i) {
    const auto &output_desc = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    uint32_t mem_type = 0U;
    if (AttrUtils::GetInt(output_desc, ATTR_OUTPUT_MEMORY_TYPE, mem_type)) {
      output_mem_types_[i] = static_cast<MemStorageType>(mem_type);
    }
    if (has_mem_type_attr && (v_memory_type[i] == static_cast<int64_t>(RT_MEMORY_HOST_SVM))) {
      output_mem_types_[i] = HOST_SVM;
    }
  }
  return SUCCESS;
}

Status NodeItem::ResolveDynamicState() {
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_FORCE_UNKNOWN_SHAPE, is_dynamic);
  GELOGD("Node name is %s, dynamic state is %d.", this->node_name.c_str(), static_cast<int32_t>(is_dynamic));
  if (!is_dynamic) {
    GE_CHK_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node, is_dynamic),
                      "[Invoke][GetNodeUnknownShapeStatus][%s(%s)] Failed to get shape status.",
                      node->GetName().c_str(), node->GetType().c_str());
  }
  return SUCCESS;
}

Status NodeItem::ResolveStaticInputsAndOutputs() {
  for (int32_t i = 0; i < num_inputs; ++i) {
    // Data has unconnected input but set by framework
    if (node_type != DATA) {
      int32_t origin_index = i;
      if (has_optional_inputs) {
        origin_index = input_desc_indices_[static_cast<size_t>(i)];
      }
      const auto &in_data_anchor = node->GetInDataAnchor(origin_index);
      GE_CHECK_NOTNULL(in_data_anchor);

      // If no node was connected to the current input anchor
      // increase num_static_input_shapes in case dead wait in ShapeInferenceState::AwaitShapesReady
      if ((in_data_anchor->GetPeerOutAnchor() == nullptr) ||
          (in_data_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr)) {
        num_static_input_shapes++;
        is_input_shape_static_.push_back(true);
        GELOGW("[%s] Peer node of input[%d] is empty", NodeName().c_str(), i);
        continue;
      }
    }
    const auto &input_desc = MutableInputDesc(i);
    GE_CHECK_NOTNULL(input_desc);
    if (input_desc->MutableShape().IsUnknownShape()) {
      is_input_shape_static_.push_back(false);
    } else {
      num_static_input_shapes++;
      is_input_shape_static_.push_back(true);
      GELOGD("[%s] The shape of input[%d] is static. shape = [%s]",
             NodeName().c_str(), i, input_desc->MutableShape().ToString().c_str());
    }
  }

  for (int32_t i = 0; i < num_outputs; ++i) {
    const auto &output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(output_desc);
    if (output_desc->MutableShape().IsUnknownShape()) {
      is_output_shape_static = false;
      break;
    }
  }

  if (is_output_shape_static) {
    GE_CHK_STATUS_RET_NOLOG(CalcOutputTensorSizes());
  }
  return SUCCESS;
}

void NodeItem::ResolveUnknownShapeType() {
  if (IsControlFlowV2Op() || (is_dynamic && (node_type == PARTITIONEDCALL))) {
    shape_inference_type = DEPEND_COMPUTE;
  } else {
    int32_t unknown_shape_type_val = 0;
    (void)AttrUtils::GetInt(op_desc, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
    shape_inference_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  }
}

void NodeItem::ResolveFftsPlusStatus() {
  const auto &owner_graph = node->GetOwnerComputeGraph();
  if ((node->GetType() != DATA) && (owner_graph != nullptr)) {
    const auto &parent_node = owner_graph->GetParentNode();
    // DATA node need drive for ffts subgraph.
    if ((parent_node != nullptr) && (parent_node->GetType() == PARTITIONEDCALL)) {
      is_ffts_sub_node_ = IsFftsGraphNode(*parent_node->GetOpDesc());
    }
  }

  // Some node of root graph is also ffts node after unfoldsubgraph
  is_ffts_sub_node_ = is_ffts_sub_node_ || IsFftsKernelNode(*node->GetOpDesc());
}

Status NodeItem::Init() {
  is_ctrl_flow_v2_op_ = ge::hybrid::IsControlFlowV2Op(node_type);
  is_ctrl_flow_op_ = kControlFlowOpTypes.count(node_type) > 0U;
  is_merge_op_ = kMergeOpTypes.count(node_type) > 0U;
  is_root_node_ = (node->GetInAllNodes().empty() || CheckIsOutLoop(node));

  GE_CHK_STATUS_RET_NOLOG(InitInputsAndOutputs());
  GE_CHK_STATUS_RET_NOLOG(ResolveDynamicState());
  ResolveFftsPlusStatus();
  ResolveUnknownShapeType();
  if (is_dynamic) {
    GE_CHK_STATUS_RET_NOLOG(ResolveStaticInputsAndOutputs());
    GE_CHK_STATUS_RET(ParseFusedSubgraph(*this),
                      "[Invoke][ParseFusedSubgraph][%s(%s)] Failed to parse fused subgraph",
                      node_name.c_str(), node_type.c_str());
  }
  skip_sufficiency_of_input_check_.clear();
  copy_mu_ = MakeShared<std::mutex>();
  GE_CHECK_NOTNULL(copy_mu_);

  return SUCCESS;
}

bool NodeItem::IsNoOp() const {
  for (int32_t i = 0; i < num_outputs; ++i) {
    const auto &tensor_desc = MutableOutputDesc(i);
    GE_RT_FALSE_CHECK_NOTNULL(tensor_desc);
    const auto &shape = tensor_desc->MutableShape();
    if (shape.IsScalar() || (shape.GetShapeSize() != 0)) {
      return false;
    }
  }
  return true;
}

bool NodeItem::IsHcclOp() const {
  return NodeExecutorManager::GetInstance().ResolveExecutorType(*this) == NodeExecutorManager::ExecutorType::HCCL;
}

//  IS_ENTER     ROOT_NODE
//        \      /
//        RESHAPE
//   The node all inputs is root node or enter node , need set enter node for loop invariant
bool NodeItem::IsInputEnterFeedNode(const NodePtr &node_tmp) const {
  if (is_root_node_ || (node_tmp->GetType() == STREAMACTIVE)) {
    return false;
  }

  for (const auto &input_node : node_tmp->GetInAllNodes()) {
    if ((kEnterOpTypes.count(NodeUtils::GetNodeType(input_node)) == 0U) &&
        (!(input_node->GetInAllNodes().empty() || CheckIsOutLoop(input_node)))) {
      return false;
    }
  }
  return true;
}

std::string NodeItem::DebugString() const {
  std::stringstream ss;
  ss << "Node: ";
  ss << "id = " << node_id;
  ss << ", name = [" << node->GetName();
  ss << "], type = " << node->GetType();
  ss << ", is_dynamic = " << (is_dynamic ? "True" : "False");
  ss << ", is_output_static = " << (is_output_shape_static ? "True" : "False");
  ss << ", unknown_shape_op_type = " << shape_inference_type;
  ss << ", stage = " << group;
  ss << ", input_start = " << input_start;
  ss << ", num_inputs = " << num_inputs;
  ss << ", output_start = " << output_start;
  ss << ", num_outputs = " << num_outputs;
  ss << ", dependent_nodes = [";
  for (const auto &dep_node : dependents_for_shape_inference) {
    ss << dep_node->GetName() << ", ";
  }
  ss << "]";
  int32_t index = 0;
  for (auto &items : outputs) {
    ss << ", output[" << index++ << "]: ";
    for (auto &item : items) {
      ss << "(" << item.second->NodeName() << ":" << item.first << "), ";
    }
  }

  return ss.str();
}

void NodeItem::SetToDynamic() {
  num_static_input_shapes = 0;
  is_dynamic = true;
  for (size_t i = 0U; i < is_input_shape_static_.size(); ++i) {
    is_input_shape_static_[i] = false;
  }
  if ((kernel_task != nullptr) && (!kernel_task->IsSupportDynamicShape())) {
    GELOGD("[%s] Dynamic shape is not supported, clear node task.", node_name.c_str());
    kernel_task = nullptr;
  }
}

GeTensorDescPtr NodeItem::DoGetInputDesc(const int32_t index) const {
  if (!has_optional_inputs) {
    return op_desc->MutableInputDesc(static_cast<uint32_t>(index));
  }

  if ((index < 0) || (index >= num_inputs)) {
    GELOGE(PARAM_INVALID, "[Check][Param:index][%s(%s)] Invalid input index, num inputs = %d, index = %d",
           node_name.c_str(), node_type.c_str(), num_inputs, index);
    REPORT_INNER_ERR_MSG("E19999", "Invalid input index, node:%s(%s) num inputs = %d, index = %d",
                       node_name.c_str(), node_type.c_str(), num_inputs, index);
    return nullptr;
  }

  return op_desc->MutableInputDesc(input_desc_indices_[static_cast<size_t>(index)]);
}

GeTensorDescPtr NodeItem::MutableInputDesc(const int32_t index) const {
  const std::lock_guard<std::mutex> lk(mu_);
  return DoGetInputDesc(index);
}

Status NodeItem::GetInputDesc(const int32_t index, GeTensorDesc &tensor_desc) const {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto input_desc = DoGetInputDesc(index);
  GE_CHECK_NOTNULL(input_desc);
  tensor_desc = *input_desc;
  return SUCCESS;
}

GeTensorDescPtr NodeItem::MutableOutputDesc(const int32_t index) const {
  const std::lock_guard<std::mutex> lk(mu_);
  return op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
}

Status NodeItem::UpdateInputDesc(const int32_t index, const GeTensorDesc &tensor_desc) const {
  const std::lock_guard<std::mutex> lk(mu_);
  const auto input_desc = DoGetInputDesc(index);
  GE_CHECK_NOTNULL(input_desc);
  *input_desc = tensor_desc;
  return SUCCESS;
}

Status NodeItem::GetCanonicalInputIndex(const uint32_t index, int32_t &canonical_index) const {
  if (!has_optional_inputs) {
    canonical_index = static_cast<int32_t>(index);
    return SUCCESS;
  }

  const auto iter = std::find(input_desc_indices_.begin(), input_desc_indices_.end(), index);
  if (iter == input_desc_indices_.end()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Param:index]input index:%u not in input_desc_indices_, check Invalid, node:%s(%s)",
           index, node_name.c_str(), node_type.c_str());
    REPORT_INNER_ERR_MSG("E19999", "input index:%u not in input_desc_indices_, check Invalid, node:%s(%s)",
                       index, node_name.c_str(), node_type.c_str());
    return INTERNAL_ERROR;
  }

  canonical_index = static_cast<int32_t>(iter - input_desc_indices_.begin());
  GELOGD("[%s] Canonicalize input index from [%u] to [%d]", node_name.c_str(), index, canonical_index);
  return SUCCESS;
}

void NodeItem::SetDataSend(NodeItem *const node_item, int32_t anchor_index) {
  if (node_item->IsMergeOp() && (IsEnterOp() || IsNextIterationOp())) {
    GELOGI("Merge node[%s] no need data ctrl from [%s : %s].", node_item->NodeName().c_str(), node->GetType().c_str(),
           node->GetName().c_str());
    return;
  }
  (void)data_send_.emplace(node_item);
  node_item->data_recv_[this] = anchor_index;
  if (is_root_node_) {
    auto &data_anchors = node_item->root_data_[this];
    (void)data_anchors.emplace(anchor_index);
  }
  // If Enter feed Not Merge, take as root Node.
  if ((IsEnterFeedNode(node) && (node_item->node_type != STREAMMERGE)) || IsInputEnterFeedNode(node)) {
    auto &data_anchors = node_item->enter_data_[this];
    (void)data_anchors.emplace(anchor_index);
  }
  GELOGI("Node[%s] will control node[%s]", NodeName().c_str(), node_item->NodeName().c_str());
}

void NodeItem::SetCtrlSend(NodeItem *const node_item, const uint32_t switch_index) {
  if (switch_index < switch_groups_.size()) {
    auto &switch_group = switch_groups_[static_cast<size_t>(switch_index)];
    (void)switch_group.emplace(node_item);
  } else {
    (void)ctrl_send_.insert(node_item);
  }

  (void)node_item->ctrl_recv_.emplace(this);
  if (is_root_node_) {
    (void)node_item->root_ctrl_.emplace(this);
  }
  // If Enter feed control signal, take as root Node.
  if (IsEnterFeedNode(node) && ((node_item->node_type != STREAMMERGE) && (node_item->node_type != STREAMACTIVE))) {
    (void)node_item->enter_ctrl_.emplace(this);
  }
  GELOGI("Node[%s] will control node[%s]", NodeName().c_str(), node_item->NodeName().c_str());
}

void NodeItem::SetMergeCtrl(NodeItem *const node_item, const uint32_t merge_index) {
  if (merge_index >= switch_groups_.size()) {
    GELOGE(FAILED, "[Check][Param][%s(%s)] group size: %zu <= merge index: %u",
           NodeName().c_str(), node_type.c_str(), switch_groups_.size(), merge_index);
    return;
  }

  // this is StreamMerge node, node_item is StreamActive node.
  auto &switch_group = switch_groups_[static_cast<size_t>(merge_index)];
  (void)switch_group.emplace(node_item);

  (void)node_item->ctrl_send_.emplace(this);
  GELOGI("Node[%s] will control node[%s]", node_item->NodeName().c_str(), NodeName().c_str());
}

size_t NodeItem::GetMergeCtrl(const uint32_t merge_index) const {
  return ((node_type == STREAMMERGE) && (merge_index < switch_groups_.size()))
         ? switch_groups_[static_cast<size_t>(merge_index)].size()
         : 0U;
}

OptionalMutexGuard::OptionalMutexGuard(std::mutex *const mutex, const std::string &name) : mu_(mutex), name_(name) {
  if (mu_ != nullptr) {
    GELOGD("lock for %s", name_.c_str());
    mu_->lock();
  }
}

OptionalMutexGuard::~OptionalMutexGuard() {
  if (mu_ != nullptr) {
    GELOGD("unlock for %s", name_.c_str());
    mu_->unlock();
    mu_ = nullptr;
  }
}
}  // namespace hybrid
}  // namespace ge
