/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"

#include <vector>
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/constant_utils.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "api/gelib/gelib.h"
#include "register/op_kernel_registry.h"
#include "graph/ge_context.h"

namespace ge {
namespace {
const int64_t kStartCallNum = 1;
const int64_t kShapeCalNum = 8;
const char *const kKernelLibName = "aicpu_ascend_kernel";
const char *const kOpsFlagClose = "0";
const char *const kPassName = "ConstantFoldingPass";
}

bool ConstantFoldingPass::NeedIgnorePass(const NodePtr &node) {
  if (folding_pass::IsNoNeedConstantFolding(node)) {
    return true;
  }
  if (AreAllOutputsEmptyShape(node->GetOpDesc())) {
    GELOGI("Current node %s is potential empty const, ignore pass.", node->GetName().c_str());
    return true;
  }
  return false;
}

bool ConstantFoldingPass::NeedFold() const {
  return need_fold_;
}

Status ConstantFoldingPass::ComputePotentialWeight(NodePtr &node, std::vector<GeTensorPtr> &outputs) {
  need_fold_ = true;
  GELOGD("Begin to perform constant folding computation on node %s.", node->GetName().c_str());
  const OpDescPtr &node_desc = node->GetOpDesc();
  auto input_nodes_2_out_anchors = OpDescUtils::GetConstInputNodeAndAnchor(*node);
  if (input_nodes_2_out_anchors.empty() || input_nodes_2_out_anchors.size() != node_desc->GetInputsSize()) {
    GELOGD("Node:%s, const input nodes size is %zu, and nodeDesc inputsSize is %zu.", node->GetName().c_str(),
           input_nodes_2_out_anchors.size(), node_desc->GetInputsSize());
    if (ConstantUtils::IsPotentialConst(node_desc)) {
      need_fold_ = false;
    }
    return NOT_CHANGED;
  }
  auto inputs = OpDescUtils::GetWeightsFromNodes(input_nodes_2_out_anchors);
  if (inputs.size() != input_nodes_2_out_anchors.size()) {
    GELOGW("Get weights from const_inputs size %zu, not match with inputs size %zu. Ignore pass.", inputs.size(),
           input_nodes_2_out_anchors.size());
    return NOT_CHANGED;
  }
  std::string memory_optimization_policy;
  (void) ge::GetContext().GetOption(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy);
  // check input nodes has potential const
  for (const auto &node_2_anchor : input_nodes_2_out_anchors) {
    if (ConstantUtils::IsPotentialConst(node_2_anchor.first->GetOpDesc())) {
      need_fold_ = false;
      break;
    }
    if (memory_optimization_policy == kMemoryPriority) {
      // in case input const node has multiple connect edge, do not fold when use memory priority policy.
      const int64_t shape_size = node_2_anchor.first->GetOpDesc()->GetOutputDesc(0).GetShape().GetShapeSize();
      if ((shape_size > kShapeCalNum) &&
          (node_2_anchor.second->GetPeerInDataNodesSize() > 1U)) {
        GELOGI("In MemoryPriority mode, ignore constant folding for node:%s when const node has multiple out edges.",
               node->GetName().c_str());
        return NOT_CHANGED;
      }
    }
  }
  // Try to run kernel on host cpu
  uint64_t start_time = GetCurrentTimestamp();
  Status compute_ret = ComputeWithHostCpuKernel(node, inputs, outputs);
  if (compute_ret == SUCCESS) {
    CollectCostTimeOfOpConstantFolding(node, start_time);
  } else {
    // If computation on AICPU is not possible, try running the host kernel within GE.
    GELOGD("Try to compute weight of %s with built-in kernel.", node->GetName().c_str());
    compute_ret = ComputeWithBuiltInKernel(node, inputs, outputs);
  }
  GELOGD("Constant folding computation for node %s (type: %s) finished, return code: %u.", node->GetName().c_str(),
         node->GetType().c_str(), compute_ret);
  return compute_ret;
}

Status ConstantFoldingPass::ComputeWithBuiltInKernel(NodePtr &node, const vector<ConstGeTensorPtr> &inputs,
                                                     std::vector<GeTensorPtr> &outputs) {
  auto op_kernel = folding_pass::GetKernelByType(node);
  if (op_kernel == nullptr) {
    GELOGD("No op kernel for node %s type %s, skip the constant folding", node->GetName().c_str(),
           node->GetType().c_str());
    return NOT_CHANGED;
  }

  // Statistic of ge constant folding kernel
  uint64_t start_time = GetCurrentTimestamp();
  auto ret = op_kernel->Compute(node->GetOpDesc(), inputs, outputs);
  CollectCostTimeOfGeConstantFolding(node, start_time);
  return ret;
}

Status ConstantFoldingPass::ComputeWithHostCpuKernel(const NodePtr &node, const vector<ConstGeTensorPtr> &inputs,
                                                     std::vector<GeTensorPtr> &outputs) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GE is not initialized or is finalized.");
    return UNSUPPORTED;
  }
  OpsKernelInfoStorePtr kernel_info = instance_ptr->OpsKernelManagerObj().GetOpsKernelInfoStore(kKernelLibName);
  if (kernel_info == nullptr) {
    GELOGE(FAILED, "[Get][OpsKernelInfoStore] %s failed", kKernelLibName);
    return UNSUPPORTED;
  }

  std::string ops_flag;
  kernel_info->opsFlagCheck(*node, ops_flag);
  if (ops_flag == kOpsFlagClose) {
    return UNSUPPORTED;
  }
  return RunOpKernel(node, inputs, outputs);
}

Status ConstantFoldingPass::RunOpKernel(const NodePtr &node,
                                        const std::vector<ConstGeTensorPtr> &inputs,
                                        std::vector<GeTensorPtr> &outputs) {
  const std::string op_type = NodeUtils::GetNodeType(node);
  auto kernel = OpKernelRegistry::GetInstance().CreateHostCpuOp(op_type);
  if (kernel == nullptr) {
    GELOGD("Op of type %s is not supported by host cpu engine", op_type.c_str());
    return UNSUPPORTED;
  }

  GELOGD("Successfully created op kernel. op type = %s", op_type.c_str());
  return HostCpuEngine::GetInstance().Run(node, *kernel, inputs, outputs);
}

const std::map<std::string, std::pair<uint64_t, uint64_t>> &ConstantFoldingPass::GetGeConstantFoldingPerfStatistic()
    const {
  return statistic_of_ge_constant_folding_;
}

const std::map<std::string, std::pair<uint64_t, uint64_t>> &ConstantFoldingPass::GetOpConstantFoldingPerfStatistic()
    const {
  return statistic_of_op_constant_folding_;
}

void ConstantFoldingPass::CollectCostTimeOfGeConstantFolding(const NodePtr &node, uint64_t start_time) {
  uint64_t cost_time = GetCurrentTimestamp() - start_time;
  if (statistic_of_ge_constant_folding_.find(node->GetType()) != statistic_of_ge_constant_folding_.end()) {
    uint64_t &cnt = statistic_of_ge_constant_folding_[node->GetType()].first;
    uint64_t &cur_cost_time = statistic_of_ge_constant_folding_[node->GetType()].second;
    cnt++;
    cur_cost_time += cost_time;
  } else {
    statistic_of_ge_constant_folding_[node->GetType()] = std::pair<uint64_t, uint64_t>(kStartCallNum, cost_time);
  }
}

void ConstantFoldingPass::CollectCostTimeOfOpConstantFolding(const NodePtr &node, uint64_t start_time) {
  if (statistic_of_op_constant_folding_.find(node->GetType()) != statistic_of_op_constant_folding_.end()) {
    uint64_t &cnt = statistic_of_op_constant_folding_[node->GetType()].first;
    uint64_t &cost_time = statistic_of_op_constant_folding_[node->GetType()].second;
    cnt++;
    cost_time += GetCurrentTimestamp() - start_time;
  } else {
    statistic_of_op_constant_folding_[node->GetType()] =
        std::pair<uint64_t, uint64_t>(kStartCallNum, GetCurrentTimestamp() - start_time);
  }
}

string ConstantFoldingPass::GetPassName() const {
  return kPassName;
}

REG_PASS_OPTION("ConstantFoldingPass").SWITCH_OPT(ge::OO_CONSTANT_FOLDING);
}  // namespace ge
