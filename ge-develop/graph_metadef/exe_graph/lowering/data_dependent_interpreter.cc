/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/data_dependent_interpreter.h"
#include "common/checker.h"
#include "graph/node.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/type/sym_dtype.h"

namespace gert {
namespace {
constexpr const ge::char_t* kUbGraph = "_original_fusion_graph";
bool IsUbFusedNode(const ge::OpDesc *const op_desc) {
  return ge::AttrUtils::HasAttr(op_desc, kUbGraph);
}
ge::graphStatus IsDataDependentByAttr(const ge::OpDesc *op_desc, const int32_t input_index, bool &is_data_dependent) {
  GE_ASSERT_NOTNULL(op_desc);
  const auto &data_dependent_inputs = op_desc->GetOpInferDepends();
  if (data_dependent_inputs.empty()) {
    is_data_dependent = false;
    return ge::GRAPH_SUCCESS;
  }
  const auto &input_name = op_desc->GetValidInputNameByIndex(static_cast<uint32_t>(input_index));
  is_data_dependent = std::find(data_dependent_inputs.cbegin(), data_dependent_inputs.cend(), input_name) !=
                      data_dependent_inputs.cend();
  return ge::GRAPH_SUCCESS;
}
ge::NodePtr FindSubgraphDataNode(const ge::ComputeGraphPtr &graph, int32_t parent_node_index) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != "Data") {
      continue;
    }
    int32_t parent_index = 0;
    if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(ge::INTERNAL_ERROR, "[Get][Attr] failed, node:[%s(%s)] attr:[%s]", node->GetName().c_str(),
             node->GetType().c_str(), ge::ATTR_NAME_PARENT_NODE_INDEX.c_str());
      REPORT_INNER_ERR_MSG("E19999", "invoke GetInt failed, node:[%s(%s)] attr:[%s]", node->GetName().c_str(),
                           node->GetType().c_str(), ge::ATTR_NAME_PARENT_NODE_INDEX.c_str());
      return nullptr;
    }
    if (parent_index == parent_node_index) {
      return node;
    }
  }
  return nullptr;
}

ge::graphStatus GetInputIrIndexByInstanceIndex(const ge::OpDescPtr &op_desc, const size_t instance_index,
                                               size_t &ir_index) {
  GE_ASSERT_NOTNULL(op_desc);
  std::map<size_t, std::pair<size_t, size_t>> ir_index_to_instance_index_pair_map;
  GE_ASSERT_GRAPH_SUCCESS(ge::GetIrInputInstanceDescRange(op_desc, ir_index_to_instance_index_pair_map));
  GE_ASSERT_TRUE(!ir_index_to_instance_index_pair_map.empty());
  ir_index = std::numeric_limits<size_t>::max();
  for (size_t i = 0U; i < op_desc->GetIrInputs().size(); ++i) {
    const auto &index_pair = ir_index_to_instance_index_pair_map[i];
    size_t ir_index_end = 0U;
    GE_ASSERT_TRUE(!ge::AddOverflow(index_pair.first, index_pair.second, ir_index_end));
    if ((instance_index >= index_pair.first) && (instance_index < ir_index_end)) {
      ir_index = i;
      GELOGD("node [%s(%s)] get ir index [%zu] successfully!", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
             ir_index);
      return ge::GRAPH_SUCCESS;
    }
  }
  GELOGW("node [%s(%s)] failed to get ir index by instance index[%zu], set ir_index to %zu", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), instance_index, ir_index);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

DataDependentInterpreter::DataDependentInterpreter(const ge::OpDescPtr &op_desc,
                                                   const gert::OpImplSpaceRegistryV2Array &space_registry)
    : op_desc_(op_desc), space_registries_v2_(space_registry) {}


ge::graphStatus DataDependentInterpreter::IsDataDependentByImplOp(const int32_t input_index,
                                                                  bool &is_data_dependent) const {
  const auto op_impl = GetOpImplFunctionsV2();
  if (op_impl == nullptr) {
    GELOGW("The node %s type %s does not registered by `IMPL_OP`", op_desc_->GetNamePtr(), op_desc_->GetType().c_str());
    is_data_dependent = false;
    // 这里产生了变更，原有实现中，如果impl找不到，并且1.0标记了任意一个输入为数据依赖，那么整个节点所有输入都会被认为是数据依赖。
    // 变更后，如果impl找不到，那么仅会返回1.0标记的输入为数据依赖。这个变更影响应该不大，验证过后，本注释可以被删除
    return ge::GRAPH_SUCCESS;
  }
  if (!op_impl->HasDataDependency()) {
    is_data_dependent = false;
    return ge::GRAPH_SUCCESS;
  }
  size_t ir_index;
  const ge::graphStatus ret = GetInputIrIndexByInstanceIndex(op_desc_, static_cast<size_t>(input_index), ir_index);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Failed to get ir index by input_index[%d] for node %s(%s).", input_index,
           op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return ge::FAILED;
  }
  is_data_dependent = op_impl->IsInputDataDependency(ir_index);
  return ge::GRAPH_SUCCESS;
}

// 此接口返回的结果表示算子是否只支持TilingDepend
// true： 算子只注册了tilingDepend，未注册DataDepend，表示算子在Infershape的时候不是dataDepend，
// 在tiling时是dataDepend
// false：其他情况
ge::graphStatus DataDependentInterpreter::IsTilingInputDataDependent(const int32_t index,
                                                                     bool &is_tiling_dependent) const {
  const auto op_impl = GetOpImplFunctionsV2();
  if (op_impl == nullptr) {
    GELOGW("The node %s type %s does not registered by `IMPL_OP`", op_desc_->GetNamePtr(), op_desc_->GetType().c_str());
    is_tiling_dependent = false;
    return ge::GRAPH_SUCCESS;
  }
  if (!op_impl->HasTilingInputDataDependency()) {
    is_tiling_dependent = false;
    return ge::GRAPH_SUCCESS;
  }

  size_t ir_index = 0UL;
  const ge::graphStatus ret = GetInputIrIndexByInstanceIndex(op_desc_, static_cast<size_t>(index), ir_index);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Failed to get ir index by input_index[%d] for node %s(%s).", index,
           op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return ge::GRAPH_FAILED;
  }
  is_tiling_dependent = op_impl->IsTilingInputDataDependency(ir_index);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DataDependentInterpreter::IsSupportTilingDependPlacement(const uint32_t placement,
                                                                         bool &is_support) const {
  const auto op_impl = GetOpImplFunctionsV2();
  if (op_impl == nullptr) {
    GELOGW("The node %s type %s does not registered by `IMPL_OP`", op_desc_->GetNamePtr(), op_desc_->GetType().c_str());
    is_support = false;
    return ge::GRAPH_SUCCESS;
  }
  is_support = op_impl->IsSupportTilingDependencyPlacement(placement);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DataDependentInterpreter::IsDataDependent(const int32_t index, bool &is_data_dependent) const {
  bool by_ir = false;
  GE_ASSERT_SUCCESS(IsDataDependentByIr(index, by_ir));
  GE_ASSERT_NOTNULL(op_desc_);
  if (!IsUbFusedNode(op_desc_.get())) {
    is_data_dependent = by_ir;
    return ge::GRAPH_SUCCESS;
  }

  bool by_ub = false;
  GE_ASSERT_SUCCESS(IsDataDependentByUbGraph(index, by_ub));
  is_data_dependent = GetByIrAndUb(by_ir, by_ub, index);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus DataDependentInterpreter::IsDataDependentByIr(int32_t index, bool &is_data_dependent) const {
  bool by_1_0 = false;
  bool by_2_0 = false;
  GE_ASSERT_SUCCESS(IsDataDependentByImplOp(index, by_2_0));
  GE_ASSERT_SUCCESS(IsDataDependentByAttr(op_desc_.get(), index, by_1_0));

  is_data_dependent = GetByIr(by_1_0, by_2_0, index);
  return ge::GRAPH_SUCCESS;
}
bool DataDependentInterpreter::GetByIr(bool by_1_0, bool by_2_0, int32_t index_for_log) const {
  if (by_1_0 == by_2_0) {
    return by_2_0;
  }
  GE_ASSERT_NOTNULL(op_desc_);
  if (by_1_0) {  // by_2_0 is false
    GELOGW(
        "The node %s type %s input index %d is interpreted data-dependent, because there is data dependent attr on the "
        "node. But the IMPL_OP does not registered as data-dependent",
        op_desc_->GetNamePtr(), op_desc_->GetTypePtr(), index_for_log);
  }
  return true;
}
ge::graphStatus DataDependentInterpreter::IsDataDependentByUbGraph(int32_t index, bool &is_data_dependent) const {
  GE_ASSERT_NOTNULL(op_desc_);
  auto ub_graph = GetUbGraph();
  GE_ASSERT_NOTNULL(ub_graph);

  const auto data_node = FindSubgraphDataNode(ub_graph, index);
  GE_ASSERT_NOTNULL(data_node, "Failed to find the data node from ub graph by index %d from node %s type %s.",
                    index, op_desc_->GetNamePtr(), op_desc_->GetTypePtr());

  is_data_dependent = false;
  for (const auto &node_and_anchor : data_node->GetOutDataNodesAndAnchors()) {
    bool node_data_dependent;
    GE_ASSERT_NOTNULL(node_and_anchor.first);
    GE_ASSERT_SUCCESS(DataDependentInterpreter(node_and_anchor.first->GetOpDesc(), space_registries_v2_)
                          .IsDataDependentByIr(node_and_anchor.second->GetIdx(), node_data_dependent));
    if (node_data_dependent) {
      is_data_dependent = true;
      break;
    }
  }

  return ge::GRAPH_SUCCESS;
}
bool DataDependentInterpreter::GetByIrAndUb(bool by_ir, bool by_ub, int32_t index_for_log) const {
  if (by_ir == by_ub) {
    return by_ir;
  }

  if (by_ir) {  // by_ub is false
    GELOGW(
        "The UB-fused node %s type %s input index %d is interpreted data-dependent. The data-dependent flag is marked "
        "by IR, but not the UB graph",
        op_desc_->GetNamePtr(), op_desc_->GetTypePtr(), index_for_log);
  }
  return true;
}
ge::ComputeGraphPtr DataDependentInterpreter::GetUbGraph() const {
  GE_ASSERT_NOTNULL(op_desc_);
  if (ub_graph_cache_ == nullptr) {
    GE_ASSERT_TRUE(ge::AttrUtils::GetGraph(op_desc_.get(), kUbGraph, ub_graph_cache_));
  }
  return ub_graph_cache_;
}

const OpImplKernelRegistry::OpImplFunctionsV2 *DataDependentInterpreter::GetOpImplFunctionsV2() const {
  GE_ASSERT_NOTNULL(op_desc_);
  std::string type;
  GE_ASSERT_SUCCESS(ge::OpTypeUtils::GetOriginalType(op_desc_, type), "Failed to get original type from %s(%s).",
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  GELOGD("GetOpImplFunctionsV2, node name %s, type %s", op_desc_->GetNamePtr(), type.c_str());

  GE_ASSERT_TRUE(space_registries_v2_.size() > static_cast<size_t>(op_desc_->GetOppImplVersion()));
  auto space_registry_v2 = space_registries_v2_[static_cast<size_t>(op_desc_->GetOppImplVersion())];
  if (space_registry_v2 == nullptr) {
    GELOGW("Attention: default registry does not exist. Tiling will be executed failed");
    return nullptr;
  }
  return space_registry_v2->GetOpImpl(type.c_str());
}
}  // namespace gert
