/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/dynamic_shape_partition.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "graph/partition/optimizer/dynamic_data_flow_partitioner_pass.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/partition/base_cluster.h"
#include "graph/utils/op_type_utils.h"
#include "graph/ge_context.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"
#include "graph/ge_local_context.h"
#include "graph/passes/pass_utils.h"
#include "api/aclgrph/option_utils.h"
#include "common/context/local_context.h"
#include "runtime/config.h"
#include "runtime/dev.h"
#include "common/ge_common/ge_types.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "graph_metadef/common/ge_common/util.h"

namespace {
const std::set<std::string> kNotilingOps = {ge::MEMCPYASYNC, ge::STREAMMERGE, ge::PARTITIONEDCALL};
constexpr size_t kDefaultMinKnownSubGraphClusterNum = 4UL;
const std::string kClusterKnownShape = "KNOWN_SHAPE";
const std::string kClusterUnknownShape = "UNKNOWN_SHAPE";
const std::string kHostCpuEngineName = "DNN_VM_HOST_CPU";
const std::string kOwnerGraphIsUnknown = "OwnerGraphIsUnknown";
constexpr int32_t kStageTypeIndex = 3;
constexpr int32_t kKnownShapeTypeIndex = 4;
constexpr int32_t kUnknownShapeTypeIndex = 5;
constexpr size_t kDefaultMinNodeNumInKnownCluster = 6UL;
constexpr int64_t kThresholdForMergeAllToUnknownGraph = -1;
constexpr int32_t kBase = 10;
const std::string kStableRdfsSort = "3";
constexpr char_t const *kOffline = "offline";
}  // namespace

namespace ge {
namespace {
Status MarkUnknownShapeOp(const ge::OpDescPtr &op, const bool is_unknown_shape) {
  GE_ASSERT_TRUE(AttrUtils::SetBool(op, ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape),
                 "[Set][Attr] %s flag to node[%s,%s] failed.", ATTR_NAME_IS_UNKNOWN_SHAPE.c_str(), op->GetNamePtr(),
                 op->GetTypePtr());
  GELOGI("Mark node[%s,%s] attr[%s] to %d.", op->GetNamePtr(), op->GetTypePtr(), ATTR_NAME_IS_UNKNOWN_SHAPE.c_str(),
         is_unknown_shape);
  return SUCCESS;
}
std::vector<std::vector<int64_t>> SplitMaxShapeStr(const std::string &str) {
  std::vector<std::vector<int64_t>> result;
  std::vector<std::string> tensor_result = StringUtils::Split(str, ';');
  for (const std::string &tensor : tensor_result) {
    std::vector<std::string> dims_result = StringUtils::Split(tensor, ',');
    std::vector<int64_t> tensor_max_shape;
    for (const std::string &dim : dims_result) {
      int64_t max_shape = std::strtol(dim.c_str(), nullptr, 10);
      tensor_max_shape.emplace_back(max_shape);
    }
    result.emplace_back(tensor_max_shape);
  }
  return result;
}

Status ParseShapeRangeAttr(const OpDescPtr &op_desc, bool &has_shape_range_attr,
                           std::vector<std::vector<int64_t>> &max_shape_list) {
  std::string max_shape_str;
  has_shape_range_attr = AttrUtils::GetStr(op_desc, ATTR_NAME_OP_MAX_SHAPE, max_shape_str);
  if (has_shape_range_attr && !max_shape_str.empty()) {
    max_shape_list = SplitMaxShapeStr(max_shape_str);
    if (max_shape_list.size() != op_desc->GetOutputsSize()) {
      GELOGE(PARAM_INVALID, "Op[%s] Invalid shape range attr, shape range size[%zu], op output size[%zu]",
             op_desc->GetName().c_str(), max_shape_list.size(), op_desc->GetOutputsSize());
      has_shape_range_attr = false;
      return PARAM_INVALID;
    }
    for (size_t i = 0UL; i < max_shape_list.size(); i++) {
      auto tensor = op_desc->GetOutputDescPtr(static_cast<uint32_t>(i));
      if (tensor == nullptr) {
        GELOGE(PARAM_INVALID, "Op[%s] Invalid shape range attr, tensor[%zu] nullptr",
               op_desc->GetName().c_str(), i);
        has_shape_range_attr = false;
        return PARAM_INVALID;
      }
      if (max_shape_list[i].size() != tensor->GetShape().GetDimNum()) {
        GELOGE(PARAM_INVALID, "Op[%s] Invalid shape range attr, tensor[%zu] input dim size[%zu], actual dim size[%zu]",
               op_desc->GetName().c_str(), i, max_shape_list[i].size(), tensor->GetShape().GetDimNum());
        has_shape_range_attr = false;
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

bool IsUnknownShape(const OpDescPtr &op_desc) {
  for (auto &out_tensor : op_desc->GetAllOutputsDescPtr()) {
    if (out_tensor != nullptr && out_tensor->GetShape().IsUnknownShape()) {
      return true;
    }
  }
  for (auto &in_tensor : op_desc->GetAllInputsDescPtr()) {
    if (in_tensor != nullptr && in_tensor->GetShape().IsUnknownShape()) {
      return true;
    }
  }
  return false;
}

bool IsExportShapeOps(const std::string &type) {
  return OpTypeUtils::IsDataNode(type) || (type == ge::MEMCPYASYNC) || (type == ge::STREAMMERGE) ||
         (type == ge::PARTITIONEDCALL) || (type == ge::CASE);
}

Status IsSupportTilingSink(gert::DataDependentInterpreter &ddi, bool &is_support) {
  is_support = false;
  std::string tiling_schedule_optimize = "0";
  (void)ge::GetContext().GetOption(ge::TILING_SCHEDULE_OPTIMIZE, tiling_schedule_optimize);
  if (tiling_schedule_optimize != "1") {
    GELOGD("Tiling sink is not supported. If support is needed, configure the tiling_schedule_optimize option.");
    return SUCCESS;
  }

  // 判断是否为离线编译场景，离线场景值根据option判断是否支持tiling下沉
  std::string build_graph_mode;
  const bool is_build_graph_offline =
      ((GetContext().GetOption(OPTION_BUILD_GRAPH_MODE, build_graph_mode) == GRAPH_SUCCESS) &&
       (build_graph_mode.compare(kOffline) == 0));
  if (!is_build_graph_offline) {
    int32_t value = 0;
    constexpr int32_t STUB_DEV_ID = 64;
    GE_CHK_RT_RET(rtGetDeviceCapability(STUB_DEV_ID, RT_MODULE_TYPE_TSCPU, FEATURE_TYPE_MODEL_TASK_UPDATE, &value));
    if (value != RT_DEV_CAP_SUPPORT) {
      GELOGD("tiling sink feature not support.");
      return SUCCESS;
    }
  }

  // 判断算子是否支持tiling下沉在aicpu上执行
  GE_ASSERT_SUCCESS(ddi.IsSupportTilingDependPlacement(static_cast<uint32_t>(gert::TilingPlacement::TILING_ON_AICPU),
                                                       is_support));
  return SUCCESS;
}
} // namespace

using Cluster = DynamicShapeCluster;
using ClusterPtr = std::shared_ptr<Cluster>;

Status DynamicShapePartitioner::MarkMemoryDiscontiguousAllocation() const {
  for (auto &node : GetRootGraph()->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc, "[Get][OpDesc] Opdesc is nullptr.");
    if (IsUnknownShape(op_desc) || JudgeUnknowShapeWithAttr(op_desc)) {
      GE_ASSERT_TRUE(AttrUtils::SetBool(*GetRootGraph(), ATTR_NAME_MEMORY_DISCONTIGUOUS_ALLOCATION, true),
                     "[Set][Attr] memory discontiguous flag on root graph:%s failed.",
                     GetRootGraph()->GetName().c_str());
      return SUCCESS;
    }
  }
  return SUCCESS;
}

void DynamicShapePartitioner::MergeClustersControlFlow() {
  for (const auto &item : control_nodes_) {
    const auto &control_node = item.second;
    for (auto it_front = control_node.begin(); it_front < control_node.end(); ++it_front) {
      for (auto it_back = it_front + 1; it_back < control_node.end(); ++it_back) {
        const auto &cluster_back = GetCluster(*it_back);
        const auto &cluster_from = GetCluster(*it_front);
        if (cluster_from == cluster_back) {
          continue;
        }
        auto merged_clusters = cluster_back->MergeAllPathFrom(cluster_from);
        GELOGI("Merge all path cluster from %lu to %lu %s.", cluster_from->Id(), cluster_back->Id(),
               ToString(merged_clusters).c_str());
        for (const auto &merged_cluster : merged_clusters) {
          for (const auto &node : merged_cluster->Nodes()) {
            SetCluster(node, cluster_back);
          }
        }
      }
    }
  }
}

Status DynamicShapePartitioner::IsSingleOpGraph(bool &is_single_op) const {
  // independent compile graph is which need to be partitioned, so set root_graph_ by independent compile graph
  (void)AttrUtils::GetBool(GetRootGraph(), ATTR_SINGLE_OP_SCENE, is_single_op);
  if (is_single_op) {
    GELOGD("Skip dynamic shape partition as in single op scene.");
    GE_ASSERT_TRUE(AttrUtils::SetBool(*GetRootGraph(), ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, false),
                   "[Set][Attr] dynamic shape partitioned flag on root graph:%s failed.",
                   GetRootGraph()->GetName().c_str());
    GE_CHK_STATUS_RET(MarkMemoryDiscontiguousAllocation(),
                      "[Set][Attr]memory discontiguous flag on root graph:%s failed",
                      GetRootGraph()->GetName().c_str());
    is_single_op = true;
    return SUCCESS;
  }
  is_single_op = false;
  return SUCCESS;
}

bool DynamicShapePartitioner::IsSubgraphMultiDims() const {
  for (const auto &subgraph : GetRootGraph()->GetAllSubgraphs()) {
    for (const auto &node : subgraph->GetDirectNode()) {
      if ((node->GetType() == DATA) && AttrUtils::HasAttr(node->GetOpDesc(), ATTR_NAME_OP_MULTI_DIMS_INPUT_DIMS)) {
        GELOGD("Subgraph %s is multi dims in root graph %s.", subgraph->GetName().c_str(),
               GetRootGraph()->GetName().c_str());
        return true;
      }
    }
  }
  return false;
}

Status DynamicShapePartitioner::IsGraphNeedUnknownShapePartition(bool &need_unknown_shape_partition) {
  bool no_need_dsp = false;
  if (AttrUtils::GetBool(GetRootGraph(), ATTR_NAME_NO_NEED_DYNAMIC_SHAPE_PARTITION, no_need_dsp) && no_need_dsp) {
    need_unknown_shape_partition = false;
    GELOGD("Skip dynamic shape partition, root graph name:%s.", GetRootGraph()->GetName().c_str());
    return SUCCESS;
  }

  GE_CHK_STATUS_RET(MarkUnknownShapeNodes(), "[Call][MarkUnknownShapeNodes] failed, root grah name:%s.",
                    GetRootGraph()->GetName().c_str());
  // 动态节点支持no tiling时:
  // 1. 若是子图分档场景，则必须要走静态图；
  // 2. 若存在动态节点，且动态节点支持no tiling，则走静态图(不包括Data及NetOutput)
  // 3. 若非上述场景，存在动态节点就应该走动态图拆分，比如Data(-1)->NetOutput(-1)，必须走动态图拆分，否则加载会出错
  if ((IsSubgraphMultiDims() || has_no_tiling_ || unknown_shape_no_tiling_nodes_.empty()) &&
      unknown_shape_nodes_.empty()) {
    GELOGD("Skip dynamic shape partition of graph %s as all nodes are known shape.", GetRootGraph()->GetName().c_str());
    GE_ASSERT_TRUE(AttrUtils::SetBool(*GetRootGraph(), ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, false),
                   "[Set][Attr] dynamic shape partitioned flag on root graph %s failed.",
                   GetRootGraph()->GetName().c_str());
    // 当前先将默认的动态调度的最大值设置为0
    std::string host_scheduling_max_threshold = "0";
    (void)ge::GetContext().GetOption(OPTION_HOST_SCHEDULING_MAX_THRESHOLD, host_scheduling_max_threshold);
    // option OPTION_HOST_SCHEDULING_MAX_THRESHOLD is check in CheckOptionValidThreshold
    int64_t max_threshold = std::strtol(host_scheduling_max_threshold.c_str(), nullptr, kBase);
    // if IsNeedTrainIteFlowCtrl is true, while insert not support V2, streamSwitch, StreamActive.....
    const bool is_need_iteration = PassUtils::IsNeedTrainIteFlowCtrl(GetRootGraph());
    if ((known_shape_nodes_.size() < static_cast<size_t>(max_threshold)) && (!has_special_node_) &&
        (!is_need_iteration)) {
      GetRootGraph()->SetGraphUnknownFlag(true);
    }
    GELOGD("graph %s, known shape node size[%zu], option max threshold[%ld], has special node status[%s],"
        "iteration status[%s].", GetRootGraph()->GetName().c_str(), known_shape_nodes_.size(), max_threshold,
        has_special_node_ ? "true" : "false", is_need_iteration ? "true" : "false");
    need_unknown_shape_partition = false;
    return SUCCESS;
  }
  GetRootGraph()->SetGraphUnknownFlag(true);
  bool is_single_op = false;
  GE_CHK_STATUS_RET(IsSingleOpGraph(is_single_op), "[Check][SingleOpGraph] failed, graph[%s].",
                    GetRootGraph()->GetName().c_str());
  if (is_single_op) {
    need_unknown_shape_partition = false;
    for (const auto &sub_node : GetRootGraph()->GetDirectNode()) {
      GELOGD("Set OwnerGraphIsUnknow attr to node[%s], graph [%s]",
             sub_node->GetName().c_str(), GetRootGraph()->GetName().c_str());
      (void)AttrUtils::SetBool(sub_node->GetOpDesc(), kOwnerGraphIsUnknown, true);
    }
    for (const auto &graph : GetRootGraph()->GetAllSubgraphs()) {
      MarkSubgraphUnknownStatus(graph);
    }
    return SUCCESS;
  }
  for (const auto &sub_node : GetRootGraph()->GetDirectNode()) {
      GELOGD("Set OwnerGraphIsUnknow attr to node[%s], graph [%s]",
             sub_node->GetName().c_str(), GetRootGraph()->GetName().c_str());
      (void)AttrUtils::SetBool(sub_node->GetOpDesc(), kOwnerGraphIsUnknown, true);
  }
  GE_ASSERT_TRUE(AttrUtils::SetBool(*GetRootGraph(), ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true),
                 "[Set][Attr] dynamic shape partitioned flag on root graph %s failed.",
                 GetRootGraph()->GetName().c_str());
  need_unknown_shape_partition = true;
  return SUCCESS;
}

void DynamicShapePartitioner::SetRootGraphUnknown() const {
  GetRootGraph()->SetGraphUnknownFlag(true);
  for (const auto &sub_node : GetRootGraph()->GetDirectNode()) {
    GELOGD("Set OwnerGraphIsUnknow attr to node[%s], graph [%s]",
           sub_node->GetName().c_str(), GetRootGraph()->GetName().c_str());
    (void)AttrUtils::SetBool(sub_node->GetOpDesc(), kOwnerGraphIsUnknown, true);
  }
  for (const auto &sub_graph : GetRootGraph()->GetAllSubgraphs()) {
    sub_graph->SetGraphUnknownFlag(true);
    for (const auto &sub_node : sub_graph->GetDirectNode()) {
      GELOGD("Set OwnerGraphIsUnknow attr to node[%s], graph [%s]",
             sub_node->GetName().c_str(), sub_graph->GetName().c_str());
      (void)AttrUtils::SetBool(sub_node->GetOpDesc(), kOwnerGraphIsUnknown, true);
    }
  }
}

bool IsUnknownGraphOnlyContainDataNetOutput(const ComputeGraphPtr &graph) {
  auto is_node_match = [](const NodePtr &node) -> bool {
    return (OpTypeUtils::IsDataNode(node->GetType())) || (node->GetType() == NETOUTPUT);
  };
  bool has_unknown_node = false;
  for (const auto &node : graph->GetAllNodes()) {
    if (!is_node_match(node)) {
      return false;
    }
    if (IsUnknownShape(node->GetOpDesc())) {
      has_unknown_node = true;
    }
  }
  return has_unknown_node;
}

Status DynamicShapePartitioner::Initialize() {
  // init static_model_ops_lower_limit
  std::string option_value;
  if (merge_known_first_) {
    static_model_ops_lower_limit_ = kDefaultMinNodeNumInKnownCluster;
  } else {
    static_model_ops_lower_limit_ = kDefaultMinKnownSubGraphClusterNum;
  }
  const graphStatus got_status = ge::GetContext().GetOption(OPTION_STATIC_MODEL_OPS_LOWER_LIMIT, option_value);
  GE_ASSERT_TRUE(!((got_status == GRAPH_SUCCESS) && (option_value.empty())), "[Get][Option] failed, %s value is empty",
                 OPTION_STATIC_MODEL_OPS_LOWER_LIMIT);
  if (option_value.empty()) {
    GELOGI("Can not get option[%s], ignore initialize.", OPTION_STATIC_MODEL_OPS_LOWER_LIMIT);
    return SUCCESS;
  }
  constexpr int64_t kMinimumOptionValue = -1;
  GE_ASSERT_SUCCESS(
      CheckValidValueRange(OPTION_STATIC_MODEL_OPS_LOWER_LIMIT, option_value, kMinimumOptionValue, INT64_MAX));
  std::istringstream val(option_value);
  val >> static_model_ops_lower_limit_;
  GELOGI("Initialize dynamic shape partitioner success, %s value is %s.", OPTION_STATIC_MODEL_OPS_LOWER_LIMIT,
         option_value.c_str());
  return SUCCESS;
}

Status DynamicShapePartitioner::GetMultiBatchIndependCompileGraphs(
    const ComputeGraphPtr &compute_graph,
    std::vector<ComputeGraphPtr> &independ_graphs) {
  bool enable_dynamic_batch = false;
  (void)ge::AttrUtils::GetBool(compute_graph, "_enable_dynamic_batch", enable_dynamic_batch);
  if (!enable_dynamic_batch) {
    GELOGI("No need to partited graph for no multi batch graph.");
    return SUCCESS;
  }
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() != CASE) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    bool is_multi_batch_case = false;
    (void)AttrUtils::GetBool(op_desc, ATTR_INSERT_BY_MBATCH, is_multi_batch_case);
    if (!is_multi_batch_case) {
      continue;
    }

    if (!op_desc->GetSubgraphInstanceNames().empty()) {
      GE_ASSERT_SUCCESS(NodeUtils::GetDirectSubgraphs(node, independ_graphs));
    }
  }
  return GRAPH_SUCCESS;
}

bool DynamicShapePartitioner::IsNeedMarkDynamicTilingDepend(const NodePtr &node) const {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  gert::OpImplSpaceRegistryV2Array space_registry_array;
  GE_ASSERT_TRUE(static_cast<size_t>(op_desc->GetOppImplVersion()) < space_registry_array.size());
  space_registry_array.at(static_cast<size_t>(op_desc->GetOppImplVersion())) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::DataDependentInterpreter ddi(op_desc, space_registry_array);
  size_t index = 0UL;
  const auto &in_anchors = node->GetAllInDataAnchors();
  for (size_t i = 0UL; i < in_anchors.size(); i++) {
    bool is_tiling_depend = false;
    const auto in_anchor = in_anchors.at(i);
    GE_ASSERT_NOTNULL(in_anchor);
    if (in_anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    GE_ASSERT_SUCCESS(ddi.IsTilingInputDataDependent(static_cast<int32_t>(index), is_tiling_depend));
    index++;
    if (is_tiling_depend) {
      // 动态shape直接打标记
      if (IsUnknownShape(node->GetOpDesc())) {
        return true;
      }
      // 输入时非const的时候打标记
      auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
      if (ge::OpDescUtils::GetInputConstData(op, static_cast<uint32_t>(i)) != nullptr) {
        continue;
      }
      GELOGD("Input[%zu] of node: %s is dyanmic tilingDependent, index: %zu",
             i, node->GetName().c_str(), index);
      return true;
    }
  }
  return false;
}

void DynamicShapePartitioner::MarkDynamicTilingDependNoe(const ComputeGraphPtr &compute_graph) const {
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (IsNeedMarkDynamicTilingDepend(node)) {
      // 此属性主要是标记此算子是动态的tilingDepend算子，可以减少后续查找注册信息的操作
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);
    }
  }
}

Status DynamicShapePartitioner::Partition() {
  GE_ASSERT_SUCCESS(Initialize());
  GE_ASSERT_NOTNULL(GetRootGraph(), "[Check][Param] Graph is nullptr.");
  MarkDynamicTilingDependNoe(GetRootGraph());
  if (ge::GetContext().GetHostExecFlag() || IsUnknownGraphOnlyContainDataNetOutput(GetRootGraph()) ||
      (static_model_ops_lower_limit_ == kThresholdForMergeAllToUnknownGraph)) {
    SetRootGraphUnknown();
    GELOGI("Graph = %s do not need dynamic shape partition, host_flag = %d, static_model_ops_lower_limit = %ld.",
           GetRootGraph()->GetName().c_str(), ge::GetContext().GetHostExecFlag(), static_model_ops_lower_limit_);
    return SUCCESS;
  }
  GE_DUMP(GetRootGraph(), "Before_DSP");

  std::vector<ComputeGraphPtr> independent_compile_graphs;
  GE_CHK_GRAPH_STATUS_RET(GetMultiBatchIndependCompileGraphs(GetRootGraph(), independent_compile_graphs),
                          "[Get][GetMultiBatchIndependCompileGraphs]failed, graph name:%s",
                          GetRootGraph()->GetName().c_str());
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::GetIndependentCompileGraphs(GetRootGraph(), independent_compile_graphs),
                          "[Get][IndependentCompileGraph]failed, graph name:%s",
                          GetRootGraph()->GetName().c_str());
  auto root_graph = GetRootGraph();  // save root_graph_ for recovery
  for (const auto &graph : independent_compile_graphs) {
    // independent compile graph is which need to be partitioned, so set root_graph_ by independent compile graph
    SetRootGraph(graph);
    GELOGD("Start dynamic shape partition graph %s.", GetRootGraph()->GetName().c_str());
    bool need_unknown_shape_partition = false;
    GE_CHK_STATUS_RET(IsGraphNeedUnknownShapePartition(need_unknown_shape_partition),
                      "[Mark][UnknownShapeGraph] failed, graph[%s]", graph->GetName().c_str());
    if (!need_unknown_shape_partition) {
      continue;
    }
    GE_CHK_STATUS_RET(CtrlEdgeTransfer(), "[Call][CtrlEdgeTransfer] failed, graph:%s.",
                      GetRootGraph()->GetName().c_str());

    auto status = PartitionImpl();
    GELOGD("%s", DebugString().c_str());
    if (status != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Call partition impl failed in dynamic shape partition, graph:%s",
                         GetRootGraph()->GetName().c_str());
      GELOGE(status, "[Call][PartitionImpl] Failed dynamic shape partition graph:%s, ret:%s",
             GetRootGraph()->GetName().c_str(), DebugString().c_str());
      ClearResource();
      return status;
    }
    GELOGD("Finish dynamic shape partition graph %s.", GetRootGraph()->GetName().c_str());
    ClearResource();
  }
  SetRootGraph(root_graph); // recovery root_graph_
  GE_DUMP(GetRootGraph(), "After_DSP");
  return SUCCESS;
}

Status DynamicShapePartitioner::CtrlEdgeTransfer() const {
  GELOGD("Do ctrl edge transfer start!");
  GE_CHECK_NOTNULL(GetRootGraph());
  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(GetRootGraph(), ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  if (!is_dynamic_shape) {
    return SUCCESS;
  }
  auto root_graph = GraphUtils::FindRootGraph(GetRootGraph());
  GE_ASSERT_NOTNULL(root_graph, "[Check][Param] root_graph not valid");
  const auto all_subgraphs = root_graph->GetAllSubgraphs();
  for (auto &subgraph : all_subgraphs) {
    for (ge::NodePtr &n : subgraph->GetDirectNode()) {
      auto op_desc = n->GetOpDesc();
      if (op_desc == nullptr) {
        continue;
      }
      auto op_type = op_desc->GetType();
      if (op_type == CONSTANT || op_type == CONSTANTOP) {
        if (n->GetInAllNodes().empty()) {
          GELOGD("[CtrlEdgeTransferPass] node [%s] in nodes is empty", n->GetName().c_str());
          continue;
        }

        GELOGD("start to tranfer ctrl edge for const node [%s]", n->GetName().c_str());

        for (auto &in_control_node : n->GetInControlNodes()) {
          GE_CHECK_NOTNULL(in_control_node);
          GE_CHECK_NOTNULL(in_control_node->GetOutControlAnchor());
          GE_CHK_GRAPH_STATUS_RET(
              ge::GraphUtils::RemoveEdge(in_control_node->GetOutControlAnchor(), n->GetInControlAnchor()),
              "[Remove][Edge] between %s and %s failed",
              in_control_node->GetOutControlAnchor()->GetOwnerNode()->GetName().c_str(), n->GetName().c_str());
          for (auto &out_node : n->GetOutNodes()) {
            if (out_node == nullptr) {
              continue;
            }
            GE_CHECK_NOTNULL(in_control_node->GetOutControlAnchor());
            GE_CHK_GRAPH_STATUS_RET(
                ge::GraphUtils::AddEdge(in_control_node->GetOutControlAnchor(), out_node->GetInControlAnchor()),
                "[Add][Edge] between %s and %s failed.",
                in_control_node->GetOutControlAnchor()->GetOwnerNode()->GetName().c_str(), out_node->GetName().c_str());
          }
        }
      }
    }
  }

  GELOGD("Do ctrl edge transfer end!");
  return SUCCESS;
}

Status DynamicShapePartitioner::ProcessUniqueClusters() {
  GE_CHK_STATUS_RET(PruneUniqueClusters(), "[Prune][Clusters] failed, graph:%s.", GetRootGraph()->GetName().c_str());
  GE_CHK_STATUS_RET(ReDynamicShapePartitioner(), "[ReDynamicShapePartitioner] failed");
  return SUCCESS;
}

Status DynamicShapePartitioner::PruneUniqueClusters() {
  auto comp_func = [](const std::shared_ptr<BaseCluster> &clu_a, const std::shared_ptr<BaseCluster> &clu_b) -> bool {
    return clu_a->Id() < clu_b->Id();
  };
  return SortUniqueClusters(comp_func);
}

Status DynamicShapePartitioner::GenerateCluster() {
  GE_CHK_STATUS_RET(MarkUnknownShapeNodes(), "[Call][MarkUnknownShapeNodes] failed, root grah name:%s.",
                    GetRootGraph()->GetName().c_str());
  GE_CHK_STATUS_RET(InitClusters(), "[Init][Clusters] failed, graph:%s.", GetRootGraph()->GetName().c_str());
  GE_CHK_STATUS_RET(MergeClusters(), "[Merge][Clusters] failed, graph:%s.", GetRootGraph()->GetName().c_str());
  GE_CHK_STATUS_RET(PruneUniqueClusters(), "[Prune][Clusters] failed, graph:%s.", GetRootGraph()->GetName().c_str());
  return SUCCESS;
}

std::string DynamicShapePartitioner::DebugString() const {
  size_t unknown = 0;
  size_t known = 0;
  size_t data = 0;
  size_t netoutput = 0;
  size_t is_inputnode = 0;
  size_t stage = 0;
  std::stringstream ss;
  ss << "All unknown shape nodes:" << std::endl;
  for (const auto &node : unknown_shape_nodes_) {
    ss << "  [" << node->GetName() << "](" << node->GetType() << ")" << std::endl;
  }
  for (const auto &unique_cluster : GetUniqueClusters()) {
    const auto cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(unique_cluster);
    if (cluster == nullptr) {
      continue;
    }
    if (cluster->IsUnknownShape()) {
      unknown++;
    } else if (cluster->IsKnownShape()) {
      known++;
    } else if (cluster->IsData()) {
      data++;
    } else if (cluster->IsNetOutput()) {
      netoutput++;
    } else if (cluster->IsInputNode()) {
      is_inputnode++;
    } else if (cluster->IsIndependent()) {
      stage++;
    }
  }
  ss << "All clusters:" << GetUniqueClusters().size() << ", data:" << data << ", known:" << known
     << ", unknown:" << unknown << ", netoutput:" << netoutput << ", is_inputnode:" << is_inputnode
     << ", stage:" << stage << std::endl;
  for (const auto &cluster : GetUniqueClusters()) {
    ss << "  " << cluster->DebugString() << std::endl;
  }
  return ss.str();
}

void DynamicShapePartitioner::ClearResource() {
  BasePartitioner::ClearResource();
  known_shape_nodes_.clear();
  unknown_shape_nodes_.clear();
  unknown_shape_no_tiling_nodes_.clear();
  control_nodes_.clear();
}

Status DynamicShapePartitioner::MarkUnknownShapeNodes() {
  for (auto &node : GetRootGraph()->GetDirectNode()) {
    GE_CHK_STATUS_RET(CollectSpreadUnknownShapeNodes(node),
                      "[Call][CollectSpreadUnknownShapeNodes] for node:%s failed.", node->GetName().c_str());
  }
  for (const auto &unknown_shape_node : unknown_shape_nodes_) {
    GELOGI("Collect unknown shape node:%s.", unknown_shape_node->GetNamePtr());
  }
  for (const auto &unknown_shape_no_tiling_node : unknown_shape_no_tiling_nodes_) {
    GELOGI("Collect unknown shape no tiling node:%s.", unknown_shape_no_tiling_node->GetNamePtr());
  }
  if (!unknown_shape_nodes_.empty() && !unknown_shape_no_tiling_nodes_.empty()) {
    GELOGW("Graph cannot support no tiling, cause [%zu] no tiling nodes and [%zu] unknown shape nodes coexist.",
           unknown_shape_no_tiling_nodes_.size(), unknown_shape_nodes_.size());
    // cannot support mixed scene now, move no tiling nodes to unknown set
    for (const auto &no_tiling_node : unknown_shape_no_tiling_nodes_) {
      RevertOpNoTilingAttr(no_tiling_node);
      unknown_shape_nodes_.insert(no_tiling_node);
    }
    unknown_shape_no_tiling_nodes_.clear();
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::InitClusters() {
  auto graph = GetRootGraph();
  GE_CHK_GRAPH_STATUS_RET(graph->TopologicalSorting(), "[Call][TopologicalSorting] failed, graph:%s.",
                          graph->GetName().c_str());
  InitClusterType();
  size_t rank = 0UL;
  for (const auto &node : graph->GetDirectNode()) {
    int32_t type_index = kDataTypeIndex;
    bool is_input = ((node->GetType() == CONSTANT) || (node->GetType() == CONSTANTOP)) && node->GetInNodes().empty();
    GE_ASSERT_NOTNULL(node->GetOpDesc(), "[Get][OpDesc] op_desc is null, graph:%s", graph->GetName().c_str());
    if (OpTypeUtils::IsDataNode(node->GetType()) && (node->GetInDataNodesSize() == 0U)) {
      // 避免refdata的回边算子单独拆分到根图, 因此增加输入数量判断
      type_index = kDataTypeIndex;
    } else if (is_input) {
      type_index = kInputNodeTypeIndex;
    } else if (node->GetType() == NETOUTPUT) {
      type_index = kNetOutputTypeIndex;
    } else if ((node->GetType() == PARTITIONEDCALL) && (node->GetOpDesc()->HasAttr(ATTR_STAGE_LEVEL))) {
      type_index = kStageTypeIndex;
    } else if (unknown_shape_nodes_.count(node) > 0U) {
      type_index = kUnknownShapeTypeIndex;
    } else {
      type_index = kKnownShapeTypeIndex;
    }
    auto cluster = MakeShared<Cluster>(rank++, type_index, node, this);
    GE_ASSERT_NOTNULL(cluster, "[New][Memory] for cluster failed.");
    RecordClusters(cluster);
    SetCluster(node, cluster);

    int64_t group_index = -1;
    if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_CONTROL_FLOW_GROUP, group_index)) {
      GELOGD("[%s] is rts control flow Op, group index: %ld", node->GetNamePtr(), group_index);
      auto &control_node = control_nodes_[group_index];
      control_node.emplace_back(node);
    }

    // Already sorted topologically, so access to the parent cluster is safe
    for (const auto &parent : node->GetInAllNodes()) {
      cluster->AddInput(GetCluster(parent));
    }
  }
  for (const auto &node : graph->GetDirectNode()) {
    GELOGD("Make cluster for node %s : %s.", node->GetNamePtr(), GetCluster(node)->DebugString().c_str());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::InitClusterType() {
  InsertIndexType(kDataTypeIndex, kClusterData);
  InsertIndexType(kInputNodeTypeIndex, kClusterInputNode);
  InsertIndexType(kNetOutputTypeIndex, kClusterNetoutput);
  InsertIndexType(kStageTypeIndex, kClusterStage);
  InsertIndexType(kKnownShapeTypeIndex, kClusterKnownShape);
  InsertIndexType(kUnknownShapeTypeIndex, kClusterUnknownShape);
  return SUCCESS;
}

Status DynamicShapePartitioner::MergeClustersUnknownShape() {
  // Merge unknown shape clusters
  std::unordered_set<std::shared_ptr<BaseCluster>> merged_clusters;
  std::unordered_set<std::shared_ptr<BaseCluster>> expired_clusters;
  for (const auto &cluster : GetOrderedClusters()) {
    if (cluster->IsIndependent()) {
      continue;
    }
    for (const auto &base_in_cluster : cluster->Inputs()) {
      const auto &in_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(base_in_cluster->shared_from_this());
      GE_ASSERT_NOTNULL(in_cluster, "[Check][ClusterInput] failed.");
      if (!in_cluster->IsUnknownShape()) {
        continue;
      }
      if (expired_clusters.count(in_cluster) != 0U) {
        continue;
      }
      auto path_clusters = cluster->MergeAllPathFrom(in_cluster);
      merged_clusters.insert(cluster);
      // path clusters will be deleted
      for (const auto &path_cluster : path_clusters) {
        expired_clusters.insert(path_cluster);
        merged_clusters.erase(path_cluster);
      }
      GELOGD("Merge all path cluster from %lu to %lu %s.", in_cluster->Id(), cluster->Id(),
             ToString(path_clusters).c_str());
    }
  }
  for (const auto &cluster : merged_clusters) {
    for (const auto &node : cluster->Nodes()) {
      SetCluster(node, cluster);
    }
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::MergeClustersInputData() {
  // Merge input clusters
  ClusterPtr cluster_pre = nullptr;
  for (const auto &cluster : GetOrderedClusters()) {
    if (!cluster->IsInputNode()) {
      continue;
    }
    if (cluster_pre != nullptr) {
      cluster_pre->Merge(cluster);
    } else {
      const auto dynamic_shape_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(cluster);
      GE_ASSERT_NOTNULL(dynamic_shape_cluster, "[Cast][Cluster] to DynamicShapeCluster failed.");
      cluster_pre = dynamic_shape_cluster;
    }
    GELOGD("Success merge input node cluster from %lu to %lu.", cluster->Id(), cluster->Id());
    for (const auto &node : cluster->Nodes()) {
      SetCluster(node, cluster_pre);
    }
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::TryMergeClusters(const ClusterFilter &cluster_filter) {
  GE_CHECK_NOTNULL(cluster_filter, "cluster filter cannot be nullptr");
  // Merge known shape clusters
  for (const auto &cluster : GetOrderedClusters()) {
    if (cluster->IsIndependent()) {
      continue;
    }
    if (cluster->IsRefVariable() && cluster->Inputs().size() == 1) {
      auto in_cluster = *(cluster->Inputs().begin());
      in_cluster->Merge(cluster);
      SetCluster(*(cluster->Nodes().begin()), in_cluster->shared_from_this());
      continue;
    }

    for (const auto &in_cluster : cluster->Inputs()) {
      if (!cluster_filter(in_cluster->shared_from_this())) {
        continue;
      }
      if (cluster->TryMerge(in_cluster->shared_from_this())) {
        GELOGD("Success merge known shape cluster from %lu to %lu.", in_cluster->Id(), cluster->Id());
        for (const auto &node : in_cluster->Nodes()) {
          SetCluster(node, cluster);
        }
      }
    }
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::ChangeSmallClusterType(const size_t threshold) const {
  for (const auto &cluster : GetOrderedClusters()) {
    const auto &dynamic_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(cluster);
    GE_ASSERT_NOTNULL(dynamic_cluster, "[Check][ClusterKnown] failed.");
    if (dynamic_cluster->IsKnownShape() && (dynamic_cluster->Nodes().size() < threshold)) {
      dynamic_cluster->SetTypeIndex(kUnknownShapeTypeIndex);
    }
    if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
      std::string nodes_name;
      for (const auto &node : dynamic_cluster->Nodes()) {
        nodes_name.append(node->GetName()).append(" ");
      }
      GELOGI("Cluster size[%zu] is less than threshold[%zu], change small cluster[%zu] to unknown, nodes(%s)",
             dynamic_cluster->Nodes().size(), threshold, cluster->Id(), nodes_name.c_str());
    }
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::MergeClusters() {
  MergeClustersControlFlow();
  std::string topo_sorting_mode;
  ge::GetContext().GetOption(ge::OPTION_TOPOSORTING_MODE, topo_sorting_mode);
  if (topo_sorting_mode == kStableRdfsSort) {
    return MergeClustersWithConsistantId();
  }
  return MergeClustersNormal();
}

Status DynamicShapePartitioner::MergeClustersInput() {
  const auto filter = [](const std::shared_ptr<BaseCluster> &cluster) {
    return cluster->IsInputNode();
  };
  const auto unique_cluster = GetUniqueClusters(filter);
  if (unique_cluster.empty()) {
    return SUCCESS;
  }
  auto merged_cluster = unique_cluster[0UL];
  GE_ASSERT_NOTNULL(merged_cluster);
  for (size_t i = 1UL; i < unique_cluster.size(); i++) {
    auto cluster = unique_cluster[i];
    GE_ASSERT_NOTNULL(cluster);
    merged_cluster->Merge(cluster);
    SetCluster(*(cluster->Nodes().begin()), merged_cluster);
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::MergeClustersWithConsistantId() {
  GE_ASSERT_SUCCESS(MergeIdConsistantCluster());
  GE_ASSERT_SUCCESS(MergeRefVariableCluster());
  GE_ASSERT_SUCCESS(MergeClustersInput());
  return SUCCESS;
}

Status DynamicShapePartitioner::MergeRefVariableCluster() {
  const auto filter = [](const std::shared_ptr<BaseCluster> &cluster) {
    return (cluster->IsRefVariable() && cluster->Inputs().size() == 1UL);
  };
  const auto unique_cluster = GetUniqueClusters(filter);
  for (size_t i = 0UL; i < unique_cluster.size(); i++) {
    GELOGI("Variable cluster: %zu should merge with its input", i);
    auto cluster = unique_cluster[i];
    GE_ASSERT_NOTNULL(cluster);
    auto in_cluster = *(cluster->Inputs().begin());
    in_cluster->Merge(cluster);
    SetCluster(*(cluster->Nodes().begin()), in_cluster->shared_from_this());
  }
  return SUCCESS;
}

void DynamicShapePartitioner::MergeClusters(std::shared_ptr<BaseCluster> &merged_cluster,
                                            std::shared_ptr<BaseCluster> &cluster) {
  merged_cluster->Merge(cluster);
  for (const auto &node : cluster->Nodes()) {
    SetCluster(node, merged_cluster);
  }
}

Status DynamicShapePartitioner::MergeIdConsistantCluster() {
  const auto filter = [](const std::shared_ptr<BaseCluster> &cluster) {
    if (cluster->IsRefVariable() && cluster->Inputs().size() == 1UL) {
      return false;
    }
    const auto dynamic_shape_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(cluster);
    if (dynamic_shape_cluster == nullptr) {
      return false;
    }
    return dynamic_shape_cluster->IsKnownShape() || dynamic_shape_cluster->IsUnknownShape();
  };
  auto unique_cluster = GetUniqueClusters(filter);
  size_t last_cluster_num = 0UL;
  size_t cluster_num = 1UL;
  for (size_t i = 1UL; i < unique_cluster.size(); i++) {
    auto &cluster = unique_cluster[i];
    GE_ASSERT_NOTNULL(cluster);
    auto last_cluster = unique_cluster[i - 1UL];
    GE_ASSERT_NOTNULL(last_cluster);
    // 当type不一致，且是最后一个算子时
    if ((last_cluster->GetTypeIndex() != cluster->GetTypeIndex()) ||
        (i == unique_cluster.size() - 1UL)) {
      // 末尾的静态算子需要加一个
      size_t static_cluster_num =
          cluster->GetTypeIndex() != kUnknownShapeTypeIndex ? cluster_num + 1UL : cluster_num;
      // 当last_cluster的type是静态，且个数小于static_model_ops_lower_limit_时，需要将其做合并
      if ((last_cluster->GetTypeIndex() == kKnownShapeTypeIndex) &&
          (static_cluster_num < static_cast<size_t>(static_model_ops_lower_limit_))) {
        // 需要调一下顺序，否则topo序会乱掉
        MergeClusters(last_cluster, cluster);
        // 静态cluster在个数少于static_model_ops_lower_limit_时，需要变成动态cluster
        last_cluster->SetTypeIndex(kUnknownShapeTypeIndex);
        cluster = last_cluster;
        cluster_num++;
        auto last_dynamic_cluster =
            i < cluster_num ? nullptr : unique_cluster[i - cluster_num];
        // 当静态cluster的前面存在动态cluster时，需要将其也合并进来
        if (last_dynamic_cluster != nullptr) {
          MergeClusters(last_dynamic_cluster, cluster);
          cluster = last_dynamic_cluster;
          cluster_num += last_cluster_num;
        }
        continue;
      }
    }
    // type一致时融合
    if (last_cluster->GetTypeIndex() == cluster->GetTypeIndex()) {
      MergeClusters(last_cluster, cluster);
      cluster = last_cluster;
      cluster_num++;
      continue;
    }
    //  不能合并
    last_cluster_num = cluster_num;
    cluster_num = 1UL;
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::MergeClustersNormal() {
  const auto filter_known = [](const std::shared_ptr<BaseCluster> &cluster) {
    const auto dynamic_shape_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(cluster);
    if (dynamic_shape_cluster == nullptr) {
      GELOGW("dynamic_shape_cluster is null, default unknown shape cluster.");
      return false;
    }
    return dynamic_shape_cluster->IsKnownShape() || dynamic_shape_cluster->IsInputNode();
  };
  const auto filter_unknown = [](const std::shared_ptr<BaseCluster> &cluster) {
    const auto dynamic_shape_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(cluster);
    if (dynamic_shape_cluster == nullptr) {
      GELOGW("dynamic_shape_cluster is null, default unknown shape cluster.");
      return true;
    }
    return dynamic_shape_cluster->IsUnknownShape();
  };
  const auto filter_known_only = [](const std::shared_ptr<BaseCluster> &cluster) {
    const auto dynamic_shape_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(cluster);
    if (dynamic_shape_cluster == nullptr) {
      GELOGW("dynamic_shape_cluster is null, default unknown shape cluster.");
      return false;
    }
    return dynamic_shape_cluster->IsKnownShape();
  };
  if (merge_known_first_) {
    GELOGD("Merge clusters with knownshape first strategy.");
    GE_CHK_STATUS_RET(TopologicalSortClusters(filter_known),
                      "[TopologicalSort][Clusters] after merge unknown shape clusters failed.");
    GE_CHK_STATUS_RET(TryMergeClusters(filter_known_only), "[Merge][KnownShapeClusters]");
    // 静态子图过小会占用流资源，且会导致动态子图较碎，回退到动态子图保证连续性
    GE_CHK_STATUS_RET(ChangeSmallClusterType(static_model_ops_lower_limit_));
    GE_CHK_STATUS_RET(MergeClustersInputData(), "[Call][MergeClustersInputData] failed.");
    GE_CHK_STATUS_RET(TopologicalSortClusters(filter_unknown),
                      "[TopologicalSort][Clusters] after merge control flow clusters failed.");
    GE_CHK_STATUS_RET(TryMergeClusters(filter_unknown), "[Merge][UnknownShapeClusters] failed.");
  } else {
    GE_CHK_STATUS_RET(TopologicalSortClusters(filter_unknown),
                      "[TopologicalSort][Clusters] after merge control flow clusters failed.");
    GE_CHK_STATUS_RET(MergeClustersUnknownShape(), "[Merge][UnknownShapeClusters] failed.");
    GE_CHK_STATUS_RET(TopologicalSortClusters(filter_known),
                      "[TopologicalSort][Clusters] after merge unknown shape clusters failed.");
    GE_CHK_STATUS_RET(TryMergeClusters(filter_known_only), "[Merge][KnownShapeClusters]");
    GE_CHK_STATUS_RET(MergeClustersInputData(), "[Call][MergeClustersInputData] failed.");
    // mall known shape graph should change to unknown shape graph
    GE_CHK_STATUS_RET(ChangeSmallClusterType(static_model_ops_lower_limit_));
    GE_CHK_STATUS_RET(TopologicalSortClusters(filter_unknown),
                      "[TopologicalSort][Clusters] after merge control flow clusters failed.");
    GE_CHK_STATUS_RET(TryMergeClusters(filter_unknown), "[Merge][UnknownShapeClusters] failed.");
  }
  return SUCCESS;
}

bool DynamicShapePartitioner::JudgeUnknowShapeWithAttr(const OpDescPtr &opdesc) const {
  bool is_forced_unknown = false;
  if (AttrUtils::GetBool(opdesc, ATTR_NAME_IS_UNKNOWN_SHAPE, is_forced_unknown) && is_forced_unknown) {
    GELOGD("Collect node %s as unknown as it was marked unknown forcibly.", opdesc->GetName().c_str());
    return true;
  }

  bool forced_unknown = false;
  if (AttrUtils::GetBool(opdesc, ATTR_NAME_FORCE_UNKNOWN_SHAPE, forced_unknown) && forced_unknown) {
    GELOGD("Collect node %s as unknown as it was marked force unknown node forcibly.", opdesc->GetNamePtr());
    return true;
  }
  return false;
}

namespace {
bool NoTilingCheckInputNodeUpdateShape(const ConstNodePtr &node) {
  auto op_desc = node->GetOpDesc();
  for (size_t i = 0; i < op_desc->GetInputsSize(); i++) {
    auto input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
    if (!input_desc->GetShape().IsUnknownShape()) {
      continue;
    }
    const auto &in_anchor = node->GetInAnchor(static_cast<int32_t>(i));
    GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
    for (auto &peer_anchor : in_anchor->GetPeerAnchorsPtr()) {
      if (peer_anchor == nullptr) {
        continue;
      }
      auto peer_node = peer_anchor->GetOwnerNode();
      GE_RT_FALSE_CHECK_NOTNULL(peer_node);
      auto peer_op_desc = peer_node->GetOpDesc();
      GE_RT_FALSE_CHECK_NOTNULL(peer_op_desc);
      if (IsExportShapeOps(peer_op_desc->GetType())) {
        GELOGD("System opType[%s], opName[%s] default support export shape.", peer_op_desc->GetType().c_str(),
               peer_op_desc->GetName().c_str());
        continue;
      }
      const std::string &peer_engine_name = peer_op_desc->GetOpEngineName();
      std::vector<std::string> update_shape_engine;
      (void)AttrUtils::GetListStr(peer_op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, update_shape_engine);
      auto it = find(update_shape_engine.begin(), update_shape_engine.end(), peer_engine_name);
      if (it == update_shape_engine.end()) {
        GELOGI("Op[%s] not support no tiling, cause parent node[%s] engine[%s] not support export shape.",
               op_desc->GetName().c_str(), peer_op_desc->GetName().c_str(), peer_engine_name.c_str());
        return false;
      }
    }
  }
  return true;
}
} // namespace

bool DynamicShapePartitioner::IsNodeSupportNoTiling(const ConstNodePtr &node) {
  auto op_desc = node->GetOpDesc();
  GE_RT_FALSE_CHECK_NOTNULL(op_desc);
  if (kNotilingOps.find(op_desc->GetType()) != kNotilingOps.end()) {
    GELOGD("System opType[%s] opName[%s] default support no tiling.", op_desc->GetType().c_str(),
           op_desc->GetName().c_str());
    has_no_tiling_ = true;
    return true;
  }
  bool no_tiling = false;
  (void)AttrUtils::GetBool(op_desc, ATTR_NAME_OP_NO_TILING, no_tiling);
  if (no_tiling) {
    GELOGD("System opType[%s] opName[%s] marked as support no tiling.", op_desc->GetType().c_str(),
           op_desc->GetName().c_str());
    has_no_tiling_ = true;
    return true;
  }

  bool is_data_or_net_output = false;
  const std::string &op_engine_name = op_desc->GetOpEngineName();
  if ((!OpTypeUtils::IsDataNode(op_desc->GetType())) && (op_desc->GetType() != NETOUTPUT)) {
    // judge engine support tiling inline
    std::vector<std::string> tiling_inline_engine;
    (void)AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_TILING_INLINE_ENGINE, tiling_inline_engine);
    auto it = find(tiling_inline_engine.begin(), tiling_inline_engine.end(), op_engine_name);
    if (it == tiling_inline_engine.end()) {
      GELOGD("Op[%s] not support no tiling, cause engine[%s] not support tiling inline.",
             op_desc->GetName().c_str(), op_engine_name.c_str());
      return false;
    }

    // judge engine support update shape after execute
    std::vector<std::string> update_shape_engine;
    (void)AttrUtils::GetListStr(op_desc, ATTR_NAME_OP_EXPORT_SHAPE_ENGINE, update_shape_engine);
    it = find(update_shape_engine.begin(), update_shape_engine.end(), op_engine_name);
    if (it == update_shape_engine.end()) {
      GELOGD("Op[%s] not support no tiling, cause engine[%s] not support update shape.",
             op_desc->GetName().c_str(), op_engine_name.c_str());
      return false;
    }
  } else {
    // Data/NetOutput节点不需要no tiling
    is_data_or_net_output = true;
  }

  // judge shape range
  std::string max_shape_str;
  bool has_shape_range_attr = AttrUtils::GetStr(op_desc, ATTR_NAME_OP_MAX_SHAPE, max_shape_str);
  if (!has_shape_range_attr) {
    for (auto &out_tensor : op_desc->GetAllOutputsDescPtr()) {
      if (!out_tensor->GetShape().IsUnknownShape()) {
        continue;
      }
      if (out_tensor->GetShape().IsUnknownDimNum()) {
        GELOGD("Op[%s] unknown dim num not support no tiling.", op_desc->GetNamePtr());
        return false;
      }
      std::vector<std::pair<int64_t, int64_t>> range;
      if (out_tensor->GetShapeRange(range) == GRAPH_FAILED || range.size() != out_tensor->GetShape().GetDimNum()) {
        GELOGD("Op[%s] not support no tiling, cause invalid shape range.", op_desc->GetName().c_str());
        return false;
      }
      for (const auto &it : range) {
        if (it.second < 0) {
          GELOGD("Op[%s] not support no tiling, cause shape range max has -1.", op_desc->GetName().c_str());
          return false;
        }
      }
    }
  }

  // judge input nodes can update shape
  if (!NoTilingCheckInputNodeUpdateShape(node)) {
    return false;
  }

  // Data/NetOutput不需要标记has_no_tiling,其他算子需要标记has_no_tiling
  has_no_tiling_ = has_no_tiling_ || (!is_data_or_net_output);
  GELOGD("Op[%s] check no tiling ok, has_no_tiling[%d], data_or_net_output[%d].", op_desc->GetName().c_str(),
         has_no_tiling_, is_data_or_net_output);
  return true;
}

void DynamicShapePartitioner::RevertOpNoTilingAttr(const NodePtr &node) const {
  auto opdesc = node->GetOpDesc();
  bool no_tiling = false;
  (void)AttrUtils::GetBool(opdesc, ATTR_NAME_OP_NO_TILING, no_tiling);
  if (!no_tiling) {
    return;
  }

  for (auto &in_tensor : opdesc->GetAllInputsDescPtr()) {
    (void)AttrUtils::SetBool(in_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, false);
  }
  for (auto &out_tensor : opdesc->GetAllOutputsDescPtr()) {
    (void)AttrUtils::SetBool(out_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, false);
  }
  (void)AttrUtils::SetBool(opdesc, ATTR_NAME_OP_NO_TILING, false);
  GELOGI("revert op[%s] no tiling attr success", opdesc->GetNamePtr());
}

Status DynamicShapePartitioner::MarkOpNoTiling(const NodePtr &node, bool no_tiling) const {
  auto opdesc = node->GetOpDesc();
  if (!no_tiling) {
    (void)AttrUtils::SetBool(opdesc, ATTR_NAME_OP_NO_TILING, no_tiling);
    return SUCCESS;
  }

  // get op max shape attr and parse to max_shape_list
  bool has_shape_range_attr = false;
  std::vector<std::vector<int64_t>> max_shape_list;
  GE_CHK_STATUS_RET(ParseShapeRangeAttr(opdesc, has_shape_range_attr, max_shape_list),
                    "[Call]node[%s] invalid input shape range. Use ';' between tensors, use ',' between dims",
                    GetRootGraph()->GetName().c_str());

  for (auto &in_tensor : opdesc->GetAllInputsDescPtr()) {
    if (in_tensor->GetShape().IsUnknownShape()) {
      (void)AttrUtils::SetBool(in_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
      GELOGD("op[%s] set in tensor no tiling", opdesc->GetName().c_str());
    }
  }
  for (size_t i = 0; i < opdesc->GetOutputsSize(); i++) {
    auto out_tensor = opdesc->MutableOutputDesc(i);
    if (out_tensor->GetShape().IsUnknownShape()) {
      (void)AttrUtils::SetBool(out_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
      GELOGD("op[%s] set out tensor[%zu] no tiling", opdesc->GetName().c_str(), i);
      if (!has_shape_range_attr) {
        continue;
      }
      (void)AttrUtils::SetListInt(out_tensor, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list[i]);
      auto anchor = node->GetOutAnchor(i);
      if (anchor == nullptr) {
        continue;
      }
      for (const auto peer_anchor : anchor->GetPeerAnchorsPtr()) {
        auto peer_node = peer_anchor->GetOwnerNode();
        auto peer_op_desc = peer_node->GetOpDesc();
        auto peer_tensor = peer_op_desc->MutableInputDesc(peer_anchor->GetIdx());
        (void)AttrUtils::SetListInt(peer_tensor, ATTR_NAME_TENSOR_MAX_SHAPE, max_shape_list[i]);
        GELOGD("No tiling set max shape for peer node[%s] tensor[%d]",
               peer_op_desc->GetName().c_str(), peer_anchor->GetIdx());
      }
    }
  }
  (void)AttrUtils::SetBool(opdesc, ATTR_NAME_OP_NO_TILING, no_tiling);
  GELOGI("mark op[%s] no tiling success", opdesc->GetName().c_str());
  return SUCCESS;
}

bool DynamicShapePartitioner::IsNodeSupportAddrRefresh(const NodePtr &node) const {
  constexpr char_t kIsSupportAddrRefresh[] = "_is_support_addr_refresh";
  bool is_support_addr_refresh = true;
  bool got = AttrUtils::GetBool(node->GetOpDescBarePtr(), kIsSupportAddrRefresh, is_support_addr_refresh);
  if (got && (!is_support_addr_refresh)) {
    GELOGI("Mark node[%s,%s] force unknown as it does not support address refresh", node->GetNamePtr(), node->GetTypePtr());
    (void)ge::AttrUtils::SetBool(node->GetOpDescBarePtr(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
    return false;
  }
  return true;
}

Status DynamicShapePartitioner::JudgeUnknownShapeForTilingDependNode(const NodePtr &node, bool &is_dynamic) const {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  gert::OpImplSpaceRegistryV2Array space_registry_array;
  GE_ASSERT_TRUE(static_cast<size_t>(op_desc->GetOppImplVersion()) < space_registry_array.size());
  space_registry_array.at(static_cast<size_t>(op_desc->GetOppImplVersion())) =
      gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::DataDependentInterpreter ddi(op_desc, space_registry_array);
  bool is_tiling_depend = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, is_tiling_depend);
  if (!is_tiling_depend) {
    is_dynamic = false;
    return SUCCESS;
  }
  // 判断是否支持host计算
  bool is_support = false;
  GE_ASSERT_SUCCESS(ddi.IsSupportTilingDependPlacement(static_cast<uint32_t>(gert::TilingPlacement::TILING_ON_HOST),
                                                       is_support));
  GE_ASSERT_TRUE(is_support);
  GE_ASSERT_SUCCESS(IsSupportTilingSink(ddi, is_support));
  if (is_support) {
    GELOGI("Node: %s is tilingDependent support tiling sink", node->GetName().c_str());
    // 为FE标记此算子，gentask阶段走框架的tiling下沉逻辑
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), "_tiling_sink_op", true);
    is_dynamic = false;
  } else {
    GELOGI("Mark node[%s,%s] force unknown as dynamic tilingDependent", node->GetNamePtr(), node->GetTypePtr());
    is_dynamic = true;
    // tilingDepend算子当tilingDepend的输入是非const算子且不支持tiling下沉时
    // 需要强制走动态，防止后续流程由于其shape是静态的而改变图的状态
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  }
  return SUCCESS;
}

bool DynamicShapePartitioner::IsSpecialNode(const OpDescPtr &op_desc) const {
  const auto type = op_desc->GetType();
  // 当前Special Node 只是用来进行对包含子图的节点进行判断，后续其他特殊的节点可以在此函数中进行扩展适配
  // StreamActive,StreamSwitch,StreamMerge,Enter,RefEnter,NextIteration,RefNextIteration V2当前未注册调度，此处屏蔽
  const bool is_special_node = ((!op_desc->GetSubgraphInstanceNames().empty()) || (type == PARTITIONEDCALL) ||
      (type == CASE) || (type == IF) || (type == WHILE) || (type == STREAMACTIVE) ||(type == STREAMSWITCH) ||
      (type == STREAMMERGE) || (type == ENTER) || (type == REFENTER) || (type == NEXTITERATION) ||
      (type == REFNEXTITERATION));
  return is_special_node;
}

Status DynamicShapePartitioner::CollectSpreadUnknownShapeNodes(NodePtr node) {
  if (unknown_shape_nodes_.count(node) > 0UL) {
    return SUCCESS;
  }
  bool is_unknown = false;
  GE_CHK_STATUS_RET(IsUnknownShapeNode(node, is_unknown), "[Call][IsUnknownShapeNode]Failed check node %s shape.",
                    node->GetName().c_str());
  if (is_unknown) {
    unknown_shape_nodes_.insert(node);
    // 与动态shape算子相连的const算子需要放在动态图中，防止多余拷贝
    for (const auto &in_node : node->GetInDataNodes()) {
      if ((in_node->GetType() == CONSTANT) || (in_node->GetType() == CONSTANTOP)) {
        unknown_shape_nodes_.insert(in_node);
      }
    }
  } else {
    // 与支持tilingDepend 功能不冲突，unknown_shape_nodes_ 如果已经赋值，则在函数IsGraphNeedUnknownShapePartition中不会使用
    // 如下的known_shape_nodes_ 和 has_special_node_ 进行功能判断
    auto opdesc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(opdesc, "[Get][OpDesc] Opdesc is nullptr.");
    known_shape_nodes_.insert(node);
    const auto is_special_node = IsSpecialNode(opdesc);
    has_special_node_ = (!has_special_node_ && is_special_node) ? true : has_special_node_;
    GELOGD("node: %s type: %s is shape known, has special node status[%s].", node->GetName().c_str(),
           opdesc->GetType().c_str(), has_special_node_ ? "true" : "false");
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::IsUnknownShapeNode(NodePtr node, bool &is_unknown) {
  auto opdesc = node->GetOpDesc();
  GE_CHECK_NOTNULL(opdesc);
  // One can set 'ATTR_NAME_IS_UNKNOWN_SHAPE=true' on node so as to forcing the node flow into the unknown subgraph,
  // ignore the actual shape.
  if (JudgeUnknowShapeWithAttr(opdesc)) {
    is_unknown = true;
    GELOGD("Mark force unknown node %s unknown as shape unknown.", node->GetName().c_str());
    return SUCCESS;
  }
  auto graph = GetRootGraph();
  GELOGD("node: %s, engine_name: %s.", node->GetNamePtr(), opdesc->GetOpEngineName().c_str());
  if (opdesc->GetOpEngineName() == kHostCpuEngineName) {
    is_unknown = true;
    GELOGD("Mark host cpu node %s unknown as host engine as it relies on the runtime scheduler for execution.",
           node->GetName().c_str());
    return SUCCESS;
  }

  bool is_shape_unknown = IsUnknownShape(opdesc);
  if (is_shape_unknown) {
    bool is_no_tiling = IsNodeSupportNoTiling(node);
    GE_CHK_STATUS_RET(MarkOpNoTiling(node, is_no_tiling), "[Call][MarkOpNoTiling] failed for node[%s].",
                      node->GetName().c_str());
    // do not set tiling attr for unknown && no tiling node's subgraph, mark unknown
    if (!is_no_tiling) {
      is_unknown = true;
      GELOGD("Mark node %s unknown as shape unknown, and cannot support no tiling.", node->GetName().c_str());
      return SUCCESS;
    }
  } else {
    GE_CHK_STATUS_RET(MarkOpNoTiling(node, false), "[Call][MarkOpNoTiling] failed for node[%s].",
                      node->GetName().c_str());
  }

  // loop subgraph to check unknown&no_tiling for known node or unknown&no_tiling node
  for (auto &subgraph_name : opdesc->GetSubgraphInstanceNames()) {
    auto subgraph = graph->GetSubgraph(subgraph_name);
    GE_ASSERT_NOTNULL(subgraph, "[Get][Subgraph] %s of node %s on root graph failed.", subgraph_name.c_str(),
                      node->GetName().c_str());
    bool is_subgraph_unknown = false;
    GE_CHK_STATUS_RET(IsUnknownShapeGraph(subgraph, is_subgraph_unknown),
                      "[Call][IsUnknownShapeGraph] Failed check subgraph %s shape of node %s.", subgraph_name.c_str(),
                      node->GetName().c_str());
    if (is_subgraph_unknown) {
      is_unknown = true;
      GE_CHK_STATUS_RET(MarkOpNoTiling(node, false), "[Call][MarkOpNoTiling] failed for node[%s].",
                        node->GetName().c_str());
      GELOGD("Mark node %s unknown as subgraph unknown, owner node is_shape_unknown[%d].",
             node->GetName().c_str(), is_shape_unknown);
      return SUCCESS;
    }
  }
  if (is_shape_unknown) {
    unknown_shape_no_tiling_nodes_.insert(node);
  }
  GE_ASSERT_SUCCESS(JudgeUnknownShapeForTilingDependNode(node, is_unknown));
  if (is_unknown || (!IsNodeSupportAddrRefresh(node))) {
    is_unknown = true;
    return SUCCESS;
  }
  is_unknown = false;
  GELOGD("Mark node %s known as %s.", node->GetName().c_str(), is_shape_unknown ? "no tiling" : "known shape");
  return SUCCESS;
}

Status DynamicShapePartitioner::IsUnknownShapeGraph(const ComputeGraphPtr &graph, bool &is_unknown) {
  for (auto &node : graph->GetDirectNode()) {
    GE_CHK_STATUS_RET(IsUnknownShapeNode(node, is_unknown),
                      "[Call][IsUnknownShapeNode]Failed check node %s shape on graph %s.", node->GetName().c_str(),
                      graph->GetName().c_str());
    if (is_unknown) {
      GELOGD("Mark graph %s unknown as contains unknown node %s.",
             graph->GetName().c_str(), node->GetName().c_str());
      return SUCCESS;
    }
  }
  return SUCCESS;
}

std::string DynamicShapePartitioner::GetSubgraphName(const BaseCluster &cluster) const {
  const auto &cast_cluster = dynamic_cast<const DynamicShapeCluster &>(cluster);
  const auto is_unknown = cast_cluster.IsUnknownShape();
  bool is_unknown_shape = is_unknown;
  bool is_input = cluster.IsInputNode();
  std::string known_name = (is_unknown_shape ? "_unknow" : "_know");
  std::string sub_graph_name_patten = (is_input ? "_input" : known_name);
  std::string sub_graph_name = GetRootGraph()->GetName() + "_sub_" + std::to_string(cluster.UniqueId()) +
                               sub_graph_name_patten;
  return sub_graph_name;
}

bool DynamicShapePartitioner::IsNeedBuildPartitionFrame(const BaseCluster &cluster) const {
  const auto &cluster_type = GetTypeByIndex(cluster.GetTypeIndex());
  return (cluster_type == kClusterUnknownShape) || (cluster_type == kClusterKnownShape) ||
         (cluster_type == kClusterInputNode);
}

void DynamicShapePartitioner::ClearReDataFlowResource() {
  for (const auto &cluster : GetUniqueClusters()) {
    cluster->Clear();
  }
  unknown_shape_nodes_.clear();
  ClearClusters();
  control_nodes_.clear();
  unknown_shape_no_tiling_nodes_.clear();
}
Status DynamicShapePartitioner::ReDynamicShapePartitioner() {
  std::vector<pair<std::string, std::shared_ptr<PartitionerPass>>> passes;
  passes.emplace_back(make_pair("DynamicDataFlowPartitionerPass", MakeShared<DynamicDataFlowPartitionerPass>()));
  while (true) {
    bool need_process = false;
    bool result = false;
    for (auto &pass : passes) {
      GE_CHECK_NOTNULL(pass.second);
      Status status = pass.second->Run(GetRootGraph(), GetSortedUniqueClusters(), result);
      need_process = need_process || result;
      if (status == SUCCESS) {
        GELOGD("Dynamic Shape RePartitioner pass %s is SUCCESS.", pass.first.c_str());
      } else {
        GELOGD("status, [Call][Run] RePartitioner pass %s failed.", pass.first.c_str());
      }
    }

    if (!need_process) {
      break;
    }
    (void)ClearReDataFlowResource();
    GE_CHK_STATUS_RET(GenerateCluster(), "[GenerateCluster] failed");
  }
  return SUCCESS;
}

std::string DynamicShapePartitioner::GetPartitionName() const {
  return "DynamicShapePartitioner";
}

Status DynamicShapePartitioner::CheckIfSubgraphUnknown(const ComputeGraphPtr &graph,
                                                       bool &is_unknown_shape) const {
  bool forced_unknown = false;
  is_unknown_shape = false;
  for (const auto &node : graph->GetDirectNode()) {
    auto desc = node->GetOpDesc();
    GE_CHK_GRAPH_STATUS_RET(ge::NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown_shape),
                            "[Get][ShapeStatus] of node[%s] failed!", node->GetName().c_str());
    if (desc->GetOpEngineName() == "DNN_VM_HOST_CPU") {
      is_unknown_shape = true;
      GELOGD("Mark host cpu node %s unknown.", node->GetName().c_str());
    }
    if (is_unknown_shape) {
      break;
    }
    if (AttrUtils::GetBool(desc, ATTR_NAME_FORCE_UNKNOWN_SHAPE, forced_unknown) && forced_unknown) {
      GELOGI("node %s was marked as force unknown shape.", node->GetName().c_str());
      is_unknown_shape = true;
      break;
    }
  }
  const auto &parent_node = graph->GetParentNode();
  if (!is_unknown_shape && parent_node != nullptr && parent_node->GetType() == PARTITIONEDCALL) {
    GE_CHK_GRAPH_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*parent_node, is_unknown_shape),
                            "[Get][ShapeStatus] of node[%s] failed!", parent_node->GetName().c_str());
  }
  return SUCCESS;
}

Status DynamicShapePartitioner::MarkSubgraphUnknownStatus(ComputeGraphPtr graph) const {
  GE_CHECK_NOTNULL(graph);
  bool is_unknown_shape = graph->GetGraphUnknownFlag();
  if (!is_unknown_shape) {
    GE_ASSERT_SUCCESS(CheckIfSubgraphUnknown(graph, is_unknown_shape));
  }
  for (const auto &sub_node : graph->GetDirectNode()) {
    GELOGD("Set OwnerGraphIsUnknown attr to node[%s]", sub_node->GetName().c_str());
    (void)AttrUtils::SetBool(sub_node->GetOpDesc(), kOwnerGraphIsUnknown, is_unknown_shape);
  }
  graph->SetGraphUnknownFlag(is_unknown_shape);
  GELOGD("mark graph [%s] unknown status success! value is %d", graph->GetName().c_str(), is_unknown_shape);
  return SUCCESS;
}

Status DynamicShapePartitioner::BuildPartitionFrame() {
  for (const auto &graph : GetRootGraph()->GetAllSubgraphs()) {
    MarkSubgraphUnknownStatus(graph);
  }
  for (const auto &cluster : sorted_unique_clusters_) {
    GE_CHK_STATUS_RET(cluster->BuildFrame(), "[Build][Frame] of cluster[%lu] failed.", cluster->Id());
  }
  return SUCCESS;
}

Status DynamicShapeCluster::BuildPartitionFrame() {
  GE_CHK_STATUS_RET(BaseCluster::BuildPartitionFrame(), "[Call][BaseCluster::BuildPartitionFrame] failed.");
  GE_CHK_STATUS_RET(SetUnknownAttr(), "[Call][SetUnknownAttr] failed.");
  return SUCCESS;
}

Status DynamicShapeCluster::SetUnknownAttr() {
  bool is_unknown_shape = IsUnknownShape();
  if (is_unknown_shape) {
    GE_ASSERT_TRUE(AttrUtils::SetBool(GetMutableSubgraph(), ATTR_NAME_MEMORY_DISCONTIGUOUS_ALLOCATION, true),
                   "[Set][Attr] memory discontiguous flag on subgraph graph:%s failed.",
                   GetMutableSubgraph()->GetName().c_str());
  }
  GetMutableSubgraph()->SetGraphUnknownFlag(is_unknown_shape);
  auto partition_op = GetPartitionNode()->GetOpDesc();
  GE_ASSERT_SUCCESS(MarkUnknownShapeOp(partition_op, is_unknown_shape));
  for (auto &node : GetMutableDirectNode()) {
    GE_ASSERT_SUCCESS(MarkUnknownShapeOp(node->GetOpDesc(), is_unknown_shape));
    (void)AttrUtils::SetBool(node->GetOpDesc(), kOwnerGraphIsUnknown, is_unknown_shape);
  }
  return SUCCESS;
}

bool DynamicShapeCluster::IsKnownShape() const {
  return GetPartitioner()->GetTypeByIndex(GetTypeIndex()) == kClusterKnownShape;
};

bool DynamicShapeCluster::IsUnknownShape() const {
  return GetPartitioner()->GetTypeByIndex(GetTypeIndex()) == kClusterUnknownShape;
}

Status DynamicShapeCluster::SetAttrToNetoutput(const OpDescPtr &op) {
  GE_ASSERT_SUCCESS(MarkUnknownShapeOp(op, IsUnknownShape()));
  return SUCCESS;
}

void DynamicShapeCluster::Merge(std::shared_ptr<BaseCluster> other) {
  BaseCluster::Merge(other);
  const auto other_dynamic_shape_cluster = std::dynamic_pointer_cast<DynamicShapeCluster>(other);
  if (!IsUnknownShape() && (other_dynamic_shape_cluster != nullptr) && other_dynamic_shape_cluster->IsUnknownShape()) {
    SetTypeIndex(other->GetTypeIndex());
    GELOGD(
        "Set cluster[%zu] type index[%d] to other cluster[%zu] type index[%d], as other is unknown and this cluster is "
        "not unknown", Id(), GetTypeIndex(), other->Id(), other->GetTypeIndex());
  }
  GELOGD("Merge cluster[%zu] to other cluster[%zu].", Id(), other->Id());
}
}  // namespace ge
