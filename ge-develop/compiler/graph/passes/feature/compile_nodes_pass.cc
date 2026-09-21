/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/compile_nodes_pass.h"

#include <utility>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "graph/op_desc.h"
#include "base/err_msg.h"

using domi::ImplyType;

namespace {
const char *const kAICPUEngineName = "DNN_VM_AICPU";
const char *const kAICPUKernelLibName = "aicpu_tf_kernel";
}  // namespace

namespace ge {
graphStatus CompileNodesPass::Run(ComputeGraphPtr graph) {
  GELOGD("[CompileNodesPass]: optimize begin.");
  if (graph == nullptr) {
    return GRAPH_SUCCESS;
  }
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if (instance == nullptr || !instance->InitFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Gelib not init before, check invalid");
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] Gelib not init before.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }
  std::unordered_map<std::string, std::vector<NodePtr>> kernel_to_compile_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    auto node_need_compile = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NEED_COMPILE, node_need_compile);
    if (!node_need_compile) {
      continue;
    }
    // collect all supported compile node
    std::string kernel_lib_name;
    auto ret = GetSupportedKernel(node, instance, kernel_lib_name);
    if (ret == GRAPH_SUCCESS) {
      auto iter = kernel_to_compile_nodes.find(kernel_lib_name);
      if (iter != kernel_to_compile_nodes.end()) {
        iter->second.emplace_back(node);
      } else {
        std::vector<NodePtr> node_vec{node};
        kernel_to_compile_nodes.insert(std::make_pair(kernel_lib_name, node_vec));
      }
    } else {
      GELOGE(GRAPH_FAILED, "[Get][SupportedKernel] for node:%s(%s) failed.", node->GetName().c_str(),
             node->GetType().c_str());
      return GRAPH_FAILED;
    }
    GELOGI("node %s %s start to call engine %s to compile.", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
           kernel_lib_name.c_str());
  }
  // compile node follow different kernel, currently only TBE kernel
  auto result = CompileNodes(instance, kernel_to_compile_nodes);
  if (result != GRAPH_SUCCESS) {
    GELOGE(result, "[Compile][Op] failed, ret:%u.", result);
    return result;
  }
  GELOGD("[CompileNodesPass]: Optimize success.");
  return GRAPH_SUCCESS;
}

graphStatus CompileNodesPass::GetSupportedKernel(const NodePtr &node, const std::shared_ptr<GELib> instance,
                                                 std::string &kernel_lib_name) const {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "[Get][OpDesc] failed, op of param node is nullptr.");
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }
  // reset op kernel lib, find supported kernel
  kernel_lib_name = op_desc->GetOpKernelLibName();
  if (kernel_lib_name.empty()) {
    (void)instance->DNNEngineManagerObj().GetDNNEngineName(node);
    kernel_lib_name = op_desc->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      REPORT_INNER_ERR_MSG("E19999", "kernel_lib_name in op:%s(%s) is empty, check invalid",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Get][OpKernelLib] for node:%s(%s) failed.", node->GetName().c_str(),
             op_desc->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
  if (kernel_info == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Find ops kernel by name:%s failed for op:%s(%s)",
                       kernel_lib_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "[Get][OpsKernelInfoStore] for op:%s failed", node->GetName().c_str());
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }

  std::map<std::string, std::string> unsupported_reasons;
  std::string unsupported_reason;
  // begin accuracy supported check
  if (!CheckAccuracySupport(kernel_info, instance, node, unsupported_reason)) {
    // if check accuracy support failed , try to go to other engine.
    GELOGD("Check Accuracy Supported return not support, node name is %s. Try to go to other engine.",
           op_desc->GetName().c_str());
    std::string kernel_name_origin = kernel_lib_name;
    OpsKernelManager &ops_kernel_manager = instance->OpsKernelManagerObj();
    auto kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
    for (auto it = kernel_map.cbegin(); it != kernel_map.cend(); ++it) {
      std::string tmp_kernel_name = it->first;
      if (tmp_kernel_name == kernel_name_origin) {
        continue;
      }
      OpsKernelInfoStorePtr tmp_kernel_info = it->second;
      if (CheckAccuracySupport(tmp_kernel_info, instance, node, unsupported_reason)) {
        kernel_lib_name = tmp_kernel_name;
        GELOGD("Find kernel lib %s support node:%s, type:%s , get kernel lib success.", tmp_kernel_name.c_str(),
               node->GetName().c_str(), op_desc->GetType().c_str());
        return GRAPH_SUCCESS;
      } else {
        unsupported_reasons.emplace(tmp_kernel_name, unsupported_reason);
      }
    }
    for (const auto &it : unsupported_reasons) {
      REPORT_PREDEFINED_ERR_MSG("EZ3002", std::vector<const char_t *>({"optype", "opskernel", "reason"}),
                         std::vector<const char_t *>({op_desc->GetType().c_str(), it.first.c_str(), it.second.c_str()}));
      GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED,
             "[Call][CheckAccuracySupport] for Op type %s of ops kernel %s is unsupported, reason:%s",
             op_desc->GetType().c_str(), it.first.c_str(), it.second.c_str());
    }

    REPORT_PREDEFINED_ERR_MSG("EZ3003", std::vector<const char_t *>({"opname", "optype"}),
                       std::vector<const char_t *>({op_desc->GetName().c_str(), op_desc->GetType().c_str()}));
    GELOGE(GRAPH_FAILED, "[Check][Param] Cannot find kernel lib support node:%s, type:%s , get kernel lib failed.",
           node->GetName().c_str(), op_desc->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool CompileNodesPass::CheckAccuracySupport(
    const OpsKernelInfoStorePtr &kernel_info, const std::shared_ptr<GELib> instance,
    const NodePtr &node, string& unsupported_reason) const {
  (void)instance;
  if (!(kernel_info->CheckAccuracySupported(node, unsupported_reason, true))) {
    return false;
  }
  return true;
}

graphStatus CompileNodesPass::CompileNodes(
    const std::shared_ptr<GELib> instance,
    std::unordered_map<std::string, std::vector<NodePtr>> &kernel_to_compile_nodes) const {
  // compile nodes, if kernel is aicpu, check support and set engine info.
  OpsKernelInfoStorePtr kernel_info;
  for (auto &kernel_nodes : kernel_to_compile_nodes) {
    kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_nodes.first);
    if (kernel_info == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Find ops kernel by name:%s failed", kernel_nodes.first.c_str());
      GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "[Get][OpsKernelInfoStore] for op %s failed", kernel_nodes.first.c_str());
      return ge::GE_GRAPH_PARAM_NULLPTR;
    }
    std::string reason;
    if (kernel_nodes.first == kAICPUKernelLibName) {
      for (auto node : kernel_nodes.second) {
        // this node will go to aicpu engine ,no need compile
        node->GetOpDesc()->SetOpEngineName(kAICPUEngineName);
        node->GetOpDesc()->SetOpKernelLibName(kAICPUKernelLibName);
        AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(ImplyType::AI_CPU));
      }
      continue;
    }
    GELOGI("The node size of compile op of %s is %zu", kernel_nodes.first.c_str(), kernel_nodes.second.size());
    auto ret = kernel_info->CompileOp(kernel_nodes.second);
    GE_ASSERT_SUCCESS(ret, "Call CompileOp failed, kernel_lib_name:%s, ret:%u", kernel_nodes.first.c_str(), ret);
  }
  return GRAPH_SUCCESS;
}

REG_PASS_OPTION("CompileNodesPass").LEVELS(OoLevel::kO0);
}  // namespace ge
