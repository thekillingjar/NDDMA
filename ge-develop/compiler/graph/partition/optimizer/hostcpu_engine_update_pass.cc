/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/optimizer/hostcpu_engine_update_pass.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph/utils/node_utils.h"
#include "common/checker.h"
#include "mmpa/mmpa_api.h"
#include "graph/ge_local_context.h"
#include "ge/ge_api_types.h"
#include "common/math/math_util.h"
#include "graph/option/optimization_option_info.h"
#include "register/optimization_option_registry.h"
#include "register/pass_option_utils.h"

namespace ge {
namespace {
constexpr uint32_t kPartitionCallSubgraphIndex = 0U;
constexpr uint32_t kDataInputIndex = 0U;
constexpr uint32_t kThenBranchIndex = 0U;
constexpr uint32_t kElseBranchIndex = 1U;
constexpr uint32_t kCondBranchIndex = 0U;
constexpr uint32_t kBodyBranchIndex = 1U;
constexpr int32_t kMaxSmallShapeElements = 8;
constexpr int32_t kIntBase = 10;
const std::string kHostCpuEngineName = "DNN_VM_HOST_CPU";
const std::string kHostCpuOpKernelLibName = "DNN_VM_HOST_CPU_OP_STORE";
const std::string kGeLocalOpKernelLibName = "DNN_VM_GE_LOCAL_OP_STORE";
const std::string kEngineNameAiCpu = "aicpu_ascend_kernel";
const std::string kEngineNameAiCpuTf = "aicpu_tf_kernel";
const std::string kEngineNameAiCore = "AIcoreEngine";
const std::string kSmallShapeHostcpu = "SmallShapeHostcpu";
const std::string kHostcpuEngineUpdatePass = "hostcpu_engine_update_pass";
const std::string kCutOffWaterFlowPassName = "HostShapeOptimizationPass";
const std::set<std::string> kHostExecOp = {SHAPE, SHAPEN, RANK, SIZE, GATHERSHAPES};
const std::set<std::string> kControlV2Types = {IF, STATELESSIF, CASE, STATELESSCASE, WHILE, STATELESSWHILE};
const std::set<std::string> kBlackList = {"RandomUniform"};
const std::set<std::string> kWhiteList = {"MapIndex"};

bool IsGelocalOp(const OpDescPtr &op_desc) {
  return op_desc->GetOpKernelLibName() == kGeLocalOpKernelLibName;
}

bool IsControlV2Op(const std::string &op_type) {
  return kControlV2Types.count(op_type) > 0U;
}

bool IsExecOnDevice(const OpDescPtr &op_desc) {
  return (op_desc->GetOpKernelLibName() == kEngineNameAiCpu) ||
         (op_desc->GetOpKernelLibName() == kEngineNameAiCpuTf) ||
         (op_desc->GetOpKernelLibName() == kEngineNameAiCore);
}

bool IsConstOp(const OpDescPtr &op_desc) {
  return (strcmp(op_desc->GetTypePtr(), CONSTANT) == 0) || (strcmp(op_desc->GetTypePtr(), CONSTANTOP) == 0);
}

bool IsAnchorOp(const OpDescPtr &op_desc) {
  if ((strcmp(op_desc->GetTypePtr(), DATA) == 0)) {
    bool is_excute_host = false;
    (void)AttrUtils::GetBool(op_desc, "_host_tensor", is_excute_host);
    if (is_excute_host) {
      GELOGI("[HostcpuEngineUpdatePass]: [%s] has been successfully added as the anchor.", op_desc->GetName().c_str());
      return true;
    }
  }
  if (op_desc->GetOpKernelLibName() == kHostCpuOpKernelLibName) {
    return true;
  }
  if (op_desc->GetOpKernelLibName() == kGeLocalOpKernelLibName) {
    return kHostExecOp.count(op_desc->GetType()) > 0U;
  }
  return false;
}

bool IsSmallEnough(ConstGeTensorDescPtr &tensor_desc, const OpDescPtr &op_desc) {
  if (kWhiteList.count(op_desc->GetType()) > 0U) {
    return true;
  }
  int64_t shape_size = tensor_desc->GetShape().GetShapeSize();
  return (shape_size >= 0) && (shape_size <= kMaxSmallShapeElements);
}

bool IsOutputSmallEnough(ConstGeTensorDescPtr &tensor_desc, const OpDescPtr &op_desc) {
  if (kWhiteList.count(op_desc->GetType()) > 0U) {
    return true;
  }
  int64_t shape_size = tensor_desc->GetShape().GetShapeSize();
  if (shape_size < 0) {
    /* 动态shape场景，以下任意一种情况满足输出足够小条件:
    1、等于一维
    2、大于一维，配置了shape range，且shape range中每一维max shape非负，累积乘小于8 */
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    (void)tensor_desc->GetShapeRange(shape_range);
    size_t dim_num = tensor_desc->GetShape().GetDimNum();
    if (!shape_range.empty() && (dim_num != 1U)) {
      int64_t max_range_size = 1;
      for (const auto& item : shape_range) {
        if (item.second < 0) {
          return false;
        }
        FMK_INT64_MULCHECK(max_range_size, item.second);
        max_range_size *= item.second;
      }
      return (max_range_size <= kMaxSmallShapeElements);
    }
    return (dim_num == 1U);
  }
  return (shape_size <= kMaxSmallShapeElements);
}

bool IsSupportHostcpu(const OpDescPtr &op_desc) {
  if (kBlackList.count(op_desc->GetType()) > 0U) {
    return false;
  }
  auto op_infos = OpsKernelManager::GetInstance().GetOpsKernelInfo(op_desc->GetType());
  for (const auto &it : op_infos) {
    if ((it.engine == kHostCpuEngineName) && (it.opKernelLib == kHostCpuOpKernelLibName)) {
      return true;
    }
  }
  return false;
}

bool CheckOutputForHostExec(const NodePtr &node, const size_t idx) {
  auto op_desc = node->GetOpDesc();
  auto output_desc = op_desc->GetOutputDescPtr(idx);
  if (output_desc == nullptr) {
    return false;
  }
  if (!IsOutputSmallEnough(output_desc, op_desc)) {
    return false;
  }
  return true;
}

bool IsDynamicGraph(const ComputeGraphPtr &graph) {
  bool is_dynamic_shape = false;
  (void)AttrUtils::GetBool(graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  if ((!is_dynamic_shape) && (!graph->GetGraphUnknownFlag())) {
    GELOGI("[HostcpuEngineUpdatePass]: graph[%s] is static shape graph, no need to update host_cpu engine",
           graph->GetName().c_str());
    return false;
  } else {
    return true;
  }
}
} // namespace

void HostcpuEngineUpdatePass::ClearNodeMapSource() {
  node_to_parent_node_.clear();
  is_node_execute_on_host_.clear();
}

bool HostcpuEngineUpdatePass::IsExecOnHost(const NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  auto it = is_node_execute_on_host_.find(node);
  if (it != is_node_execute_on_host_.end()) {
    if (it->second) {
      return true;
    }
    // CheckAndMarkHostExec modify engineName to DNN_VM_HOST_CPU_OP_STORE
    return (host_exe_ops_.count(node) > 0U);
  }
  bool ret = IsAnchorOp(op_desc) || IsConstOp(op_desc) || host_exe_ops_.count(node) > 0U;
  is_node_execute_on_host_[node] = ret;
  return ret;
}

NodePtr HostcpuEngineUpdatePass::GetParentNode(const NodePtr &node) {
  auto it = node_to_parent_node_.find(node);
  NodePtr in_node = nullptr;
  if (it != node_to_parent_node_.end()) {
    in_node = it->second;
  } else {
    in_node = NodeUtils::GetParentInput(node);
    node_to_parent_node_[node] = in_node;
  }
  return in_node;
}

bool HostcpuEngineUpdatePass::CheckInputForHostExec(const NodePtr &node, const size_t idx) {
  auto op_desc = node->GetOpDesc();
  auto input_desc = op_desc->GetInputDescPtr(idx);
  if (input_desc == nullptr) {
    return false;
  }
  if (!IsSmallEnough(input_desc, op_desc)) {
    return false;
  }
  auto peer_node = NodeUtils::GetInDataNodeByIndex(*node, idx);
  if (peer_node == nullptr) {
    return false;
  }
  auto peer_op_desc = peer_node->GetOpDesc();
  if (peer_op_desc == nullptr) {
    return false;
  }
  if (!IsExecOnHost(peer_node)) {
    return false;
  }
  return true;
}

bool HostcpuEngineUpdatePass::CheckAndMarkHostExec(const NodePtr &node, NodeEngineMap &node_atomic_engine_map,
                                                   NodeEngineMap &node_composite_engine_map) {
  if (IsExecOnHost(node)) {
    GELOGD("[HostcpuEngineUpdatePass]: node[%s] execute on host", node->GetName().c_str());
    return true;
  }
  auto op_desc = node->GetOpDesc();
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    if (!CheckOutputForHostExec(node, i)) {
      GELOGD("[HostcpuEngineUpdatePass]: node[%s] outIdx[%zu] not execute on host", op_desc->GetName().c_str(), i);
      return false;
    }
  }
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    if (!CheckInputForHostExec(node, i)) {
      GELOGD("[HostcpuEngineUpdatePass]: node[%s] inputIdx[%zu] not execute on host", op_desc->GetName().c_str(), i);
      return false;
    }
  }
  if (IsGelocalOp(op_desc)) {
    host_exe_ops_.insert(node);
    return true;
  }
  if (IsSupportHostcpu(op_desc) && IsExecOnDevice(op_desc)) {
    op_desc->SetOpEngineName(kHostCpuEngineName);
    op_desc->SetOpKernelLibName(kHostCpuOpKernelLibName);
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_ENGINE_NAME_FOR_LX, kHostCpuEngineName);
    (void)AttrUtils::SetStr(op_desc, ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, kHostCpuOpKernelLibName);
    (void)AttrUtils::SetBool(op_desc, kSmallShapeHostcpu, true);
    node_atomic_engine_map[node] = kHostCpuEngineName;
    node_composite_engine_map[node] = kHostCpuEngineName;
    GELOGI("[HostcpuEngineUpdatePass]: Set OpKernelLibName %s and OpEngineName %s to %s", kHostCpuOpKernelLibName,
           kHostCpuEngineName, op_desc->GetName().c_str());
    host_exe_ops_.insert(node);
    return true;
  }
  return false;
}

Status HostcpuEngineUpdatePass::FindHostDataOfSubgraph(const ComputeGraphPtr &graph, std::deque<NodePtr> &q) {
  if (!IsDynamicGraph(graph)) {
    return SUCCESS;
  }
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (strcmp(node->GetTypePtr(), DATA) == 0) {
      NodePtr in_node = GetParentNode(node);
      if ((in_node != nullptr) && IsExecOnHost(in_node)) {
        if (host_exe_ops_.count(node) > 0U) {
          continue;
        }
        GELOGD("[HostcpuEngineUpdatePass]: graph[%s] input[%s] is get host data[%s] cross subgraph",
               graph->GetName().c_str(), node->GetName().c_str(), in_node->GetName().c_str());
        q.emplace_back(node);
        host_exe_ops_.insert(node);
      }
    }
  }
  return SUCCESS;
}

Status HostcpuEngineUpdatePass::FindHostDataOfPartitionCall(const ComputeGraphPtr &graph, std::deque<NodePtr> &q) {
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (strcmp(node->GetTypePtr(), DATA) == 0) {
      NodePtr in_node = nullptr;
      GE_ASSERT_SUCCESS(NodeUtils::GetInNodeCrossPartionedCallNode(node, kDataInputIndex, in_node));
      if (in_node == nullptr) {
        in_node = GetParentNode(node);
      }
      if ((in_node != nullptr) && IsExecOnHost(in_node)) {
        if (host_exe_ops_.count(node) == 0U) {
          GELOGD("[HostcpuEngineUpdatePass]: graph[%s] input[%s] is get host data[%s] cross partitioncall",
                 graph->GetName().c_str(), node->GetName().c_str(), in_node->GetName().c_str());
          q.emplace_back(node);
          host_exe_ops_.insert(node);
        }
      }
    }
  }
  return SUCCESS;
}

Status HostcpuEngineUpdatePass::FindHostInputOfControlV2Node(const NodePtr &node, std::deque<NodePtr> &q) {
  auto node_type = node->GetTypePtr();
  if ((strcmp(node_type, IF) == 0) || (strcmp(node_type, STATELESSIF) == 0)) {
    GELOGD("[HostcpuEngineUpdatePass]: find host input of IfOp[%s] subgraph.", node->GetName().c_str());
    const auto &then_subgraph = NodeUtils::GetSubgraph(*node, kThenBranchIndex);
    const auto &else_subgraph = NodeUtils::GetSubgraph(*node, kElseBranchIndex);
    GE_CHECK_NOTNULL(then_subgraph);
    GE_CHECK_NOTNULL(else_subgraph);
    GE_ASSERT_SUCCESS(FindHostDataOfSubgraph(then_subgraph, q));
    GE_ASSERT_SUCCESS(FindHostDataOfSubgraph(else_subgraph, q));
  } else if ((strcmp(node_type, CASE) == 0) || (strcmp(node_type, STATELESSCASE) == 0)) {
    GELOGD("[HostcpuEngineUpdatePass]: find host input of CaseOp[%s] subgraph.", node->GetName().c_str());
    const size_t num_subgraphs = node->GetOpDesc()->GetSubgraphInstanceNames().size();
    for (size_t i = 0U; i < num_subgraphs; ++i) {
      const auto sub_graph = NodeUtils::GetSubgraph(*node, i);
      GE_CHECK_NOTNULL(sub_graph);
      GE_ASSERT_SUCCESS(FindHostDataOfSubgraph(sub_graph, q));
    }
  } else if ((strcmp(node_type, WHILE) == 0) || (strcmp(node_type, STATELESSWHILE) == 0)) {
    GELOGD("[HostcpuEngineUpdatePass]: find host input of WhileOp[%s] subgraph.", node->GetName().c_str());
    const auto cond_subgraph = NodeUtils::GetSubgraph(*node, kCondBranchIndex);
    const auto body_subgraph = NodeUtils::GetSubgraph(*node, kBodyBranchIndex);
    GE_CHECK_NOTNULL(cond_subgraph);
    GE_CHECK_NOTNULL(body_subgraph);
    GE_ASSERT_SUCCESS(FindHostDataOfSubgraph(cond_subgraph, q));
    GE_ASSERT_SUCCESS(FindHostDataOfSubgraph(body_subgraph, q));
  }
  return SUCCESS;
}

Status HostcpuEngineUpdatePass::FindAndMarkHostCpuNode(std::deque<NodePtr> &q, NodeEngineMap &node_atomic_engine_map,
                                                       NodeEngineMap &node_composite_engine_map) {
  while (!q.empty()) {
    auto top = q.front();
    GE_ASSERT_NOTNULL(top);
    auto top_op_desc = top->GetOpDesc();
    GE_ASSERT_NOTNULL(top_op_desc);
    q.pop_front();
    if (IsControlV2Op(top_op_desc->GetTypePtr())) {
      GE_ASSERT_SUCCESS(FindHostInputOfControlV2Node(top, q));
    } else if (CheckAndMarkHostExec(top, node_atomic_engine_map, node_composite_engine_map)) {
      auto output_nodes = top->GetOutDataNodes();
      for (auto output_node : output_nodes) {
        GE_ASSERT_NOTNULL(output_node);
        if (host_exe_ops_.count(output_node) == 0U) {
          GELOGD("[HostcpuEngineUpdatePass]: top_node[%s], output_node[%s], output_nodes size is %d",
                 top->GetName().c_str(), output_node->GetName().c_str(), output_nodes.size());
          q.emplace_back(output_node);
        }
      }
    }
  }
  return SUCCESS;
}

Status HostcpuEngineUpdatePass::UpdateHostcpuEngine(const ComputeGraphPtr &graph, NodeEngineMap &node_atomic_engine_map,
                                                    NodeEngineMap &node_composite_engine_map, bool is_partition_call) {
  if (ge::GraphUtils::IsSingleOpScene(graph)) {
    GELOGI("[HostcpuEngineUpdatePass]: does not support single op scene");
    return SUCCESS;
  }
  if (!IsDynamicGraph(graph)) {
    return SUCCESS;
  }
  std::deque<NodePtr> q;
  if (is_partition_call) {
    GE_ASSERT_SUCCESS(FindHostDataOfPartitionCall(graph, q));
  }
  for (const auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (strcmp(op_desc->GetTypePtr(), PARTITIONEDCALL) == 0) {
      const auto &sub_graph = NodeUtils::GetSubgraph(*node, kPartitionCallSubgraphIndex);
      GE_CHECK_NOTNULL(sub_graph);
      GE_ASSERT_SUCCESS(UpdateHostcpuEngine(sub_graph, node_atomic_engine_map, node_composite_engine_map, true));
    }
    if (IsAnchorOp(op_desc)) {
      GELOGD("[HostcpuEngineUpdatePass]: graph[%s] get anchor op %s", graph->GetName().c_str(),
             op_desc->GetName().c_str());
      q.emplace_back(node);
      host_exe_ops_.insert(node);
    }
  }
  GE_ASSERT_SUCCESS(FindAndMarkHostCpuNode(q, node_atomic_engine_map, node_composite_engine_map));
  return SUCCESS;
}

Status HostcpuEngineUpdatePass::Run(const ComputeGraphPtr &graph,
                                    NodeEngineMap &node_atomic_engine_map,
                                    NodeEngineMap &node_composite_engine_map) {
  /**
   * User-specified pass name which takes precedence over optimization option levels (eg., O1, O3).
   * 
   * Previous logic:
   *   - O0/O1: option is disabled, so this function pass is disabled.
   *   - O2/O3: option is enabled, so this function pass is enabled if the runtime2.0 switch is on.
   * 
   * Updated logic:
   *   - If the user explicitly configures the 'HostShapeOptimizationPass' pass, 
   *     this function pass will be enabled regardless of Oo level.
   *   - If the user does not configure the 'HostShapeOptimizationPass' pass:
   *     - Under O0/O1: this function pass remains disabled
   *     - Under O2/O3: the default behavior enables this function pass
   */
  bool cut_off_water_flow_enable = false;
  Status status = PassOptionUtils::CheckIsPassEnabled(kCutOffWaterFlowPassName, cut_off_water_flow_enable);
  if (status == SUCCESS && cut_off_water_flow_enable) {
    GELOGI("Hostcpu engine update open when cut off water flow flag is true.");
  } else if (status == SUCCESS && !cut_off_water_flow_enable) {
    GELOGI("Hostcpu engine update close when cut off water flow flag is false.");
    return SUCCESS;
  } else {
    std::string opt_value;
    status = GetThreadLocalContext().GetOo().GetValue(kHostcpuEngineUpdatePass, opt_value);
    GELOGI("Get option[%s], opt_value[%s], status[%u].",
            kHostcpuEngineUpdatePass.c_str(), opt_value.c_str(), status);
    // 没有注册或者获取OoLevel与注册时默认值不匹配（OoLevel为0/1），则不支持hostcpu engine update pass
    if (opt_value == "false") {
      GELOGI("Hostcpu engine update close in current level.");
      return SUCCESS;
    }
    int64_t enable_runtime2 = -1;
    const char_t *runtime2_env = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ENABLE_RUNTIME_V2, runtime2_env);
    if (runtime2_env != nullptr) {
      enable_runtime2 = std::strtol(runtime2_env, nullptr, kIntBase);
      GELOGI("[HostcpuEngineUpdatePass]: enable_runtime2 is %lld.", enable_runtime2);
    }

    // 0 means that runtime2 is closed, host_cpu is not enabled
    if (enable_runtime2 == 0) {
      return SUCCESS;
    }
  }

  GE_ASSERT_NOTNULL(graph);
  GELOGI("[HostcpuEngineUpdatePass]: graph[%s] optimize start.", graph->GetName().c_str());
  GE_MAKE_GUARD(clear_node_map, [this]() { ClearNodeMapSource(); });
  GE_ASSERT_SUCCESS(UpdateHostcpuEngine(graph, node_atomic_engine_map, node_composite_engine_map));
  GELOGI("[HostcpuEngineUpdatePass]: graph[%s] optimize succ, host ops size=%u.",
         graph->GetName().c_str(), host_exe_ops_.size());
  return SUCCESS;
}

// The associated option which registered the 'hostcpu_engine_update_pass' option for GE.
// Did not directly register pass name to GE.
REG_OPTION(kHostcpuEngineUpdatePass)
    .LEVELS(ge::OoLevel::kO0)
    .DEFAULT_VALUES({{ge::OoLevel::kO0, "false"}, {ge::OoLevel::kO1, "false"},
                     {ge::OoLevel::kO2, "true"}, {ge::OoLevel::kO3, "true"}});
}  // namespace ge
