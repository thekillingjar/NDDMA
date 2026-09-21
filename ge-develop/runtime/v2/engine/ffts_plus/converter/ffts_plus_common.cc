/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ffts_plus_common.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph/utils/graph_utils.h"
#include "engine/aicore/converter/bg_kernel_launch.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "exe_graph/lowering/value_holder_utils.h"
#include "engine/aicore/converter/aicore_node_converter.h"
namespace gert {
FFTSNodeCalculaterRegistry::NodeCalculater GetNodeCalculater(const ge::NodePtr &node) {
  auto calc_func = ge::AttrUtils::GetStr(node->GetOpDesc(), ge::kAttrCalcArgsSizeFunc);
  if (calc_func != nullptr) {
    auto func = FFTSNodeCalculaterRegistry::GetInstance().FindNodeCalculater(*calc_func);
    if (func == nullptr) {
      GELOGE(ge::FAILED, "Get null calculater of node[%s] by attr.", node->GetName().c_str());
      return nullptr;
    }
    return func;
  }
  auto func = FFTSNodeCalculaterRegistry::GetInstance().FindNodeCalculater(node->GetOpDesc()->GetOpKernelLibName());
  if (func != nullptr) {
    return func;
  }
  GELOGE(ge::FAILED, "Failed to find the calculater for node %s type %s", node->GetName().c_str(),
         node->GetType().c_str());
  return nullptr;
}

ge::Status CreateMemoryGuard(FFTSAllMemPara &all_mem_para) {
  size_t total_size = 0;
  auto guard_holder = ContinuousVector::Create<MemGuard>(all_mem_para.node_to_args_para.size(), total_size);
  if (guard_holder == nullptr) {
    GELOGE(ge::FAILED, "Create memory guard vector failed.");
    return ge::FAILED;
  }
  auto vec = reinterpret_cast<ContinuousVector *>(guard_holder.get());
  vec->SetSize(0);
  all_mem_para.mem_guarder = bg::ValueHolder::CreateConst(vec, total_size);
  FE_ASSERT_NOTNULL(all_mem_para.mem_guarder);
  return ge::SUCCESS;
}

bg::ValueHolderPtr CreateNodeMemParam(const ge::NodePtr &node, FFTSAllMemPara &all_mem_para, std::string key) {
  if (IsNoNeedCalcSize(node)) {
    GELOGD("Node[%s] does not need to calculate size.", node->GetName().c_str());
    return nullptr;
  }
  std::string para_key = key.empty() ? node->GetName() : key;
  const auto iter = all_mem_para.node_to_args_para.find(para_key);
  if (iter == all_mem_para.node_to_args_para.end()) {
    GELOGE(ge::FAILED, "Node[%s] not find args para.", node->GetName().c_str());
    return nullptr;
  }
  auto node_ori_para = bg::ValueHolder::CreateConst(&iter->second, sizeof(iter->second));
  if (node_ori_para == nullptr) {
    GELOGE(ge::FAILED, "Node[%s] create para const failed.", node->GetName().c_str());
    return nullptr;
  }
  auto holder = bg::ValueHolder::SetScopedCurrentComputeNode(node);
  auto ret = bg::ValueHolder::CreateDataOutput("NodeMemParaAssign",
      {node_ori_para, iter->second.args_data, all_mem_para.dev_addr_base, all_mem_para.host_addr_base},
      static_cast<size_t>(MemParaOutKey::kNUM));
  CHECK_HOLDERS_ALL_OK_RET(ret, static_cast<size_t>(MemParaOutKey::kNUM), return nullptr);
  bg::ValueHolder::AddDependency(all_mem_para.mem_guarder, ret[static_cast<size_t>(MemParaOutKey::MEM_GUARD)]);
  ret[static_cast<size_t>(MemParaOutKey::MEM_GUARD)]->RefFrom(all_mem_para.mem_guarder);
  GELOGD("Node [%s] successfully created with memory parameters.", node->GetNamePtr());
  return ret[static_cast<size_t>(MemParaOutKey::NODE_PARA)];
}

std::vector<bg::ValueHolderPtr> FFTSTaskAndArgsLaunch(FFTSLuanchArg launch_arg, FFTSAllMemPara &all_mem_para,
                                                      std::vector<bg::ValueHolderPtr> &task_info_vec) {
  if (launch_arg.need_launch == nullptr) {
    uint32_t need_flag = 1U;
    launch_arg.need_launch = bg::ValueHolder::CreateConst(&need_flag, sizeof(need_flag));
  }
  auto h2d_ret = bg::ValueHolder::CreateVoid<bg::ValueHolder>("FFTSTaskAndArgsCopy",
      {launch_arg.global_data->GetStream(), all_mem_para.dev_addr_base,
       all_mem_para.host_addr_base, all_mem_para.mem_guarder, launch_arg.need_launch});
  auto task_info_para = task_info_vec[static_cast<size_t>(TaskPreOutKey::NODE_PARA)];
  DfxExeArg dfx_exe_arg = GetOpDfxExeArg(launch_arg.node);
  auto dfx_holder = bg::ValueHolder::CreateConst(&dfx_exe_arg, sizeof(dfx_exe_arg));
  auto launch_ret = bg::LaunchFFTSPlusTaskNoCopy(launch_arg.global_data->GetStream(), task_info_para,
                                                 launch_arg.need_launch, dfx_holder, launch_arg.workspaces_addr);
  if (launch_ret == nullptr) {
    GELOGE(ge::FAILED, "Lowing launch ffts task failed with copy flag[%d].", launch_arg.do_copy);
    return {};
  }
  bg::ValueHolder::AddDependency(h2d_ret, launch_ret);

  return {h2d_ret, launch_ret};
}

ge::Status LoweringGraphPostProc(const LowerResult *graph_result, const std::vector<bg::ValueHolderPtr> &task_ret,
    const std::vector<bg::ValueHolderPtr> &free_vec, const std::vector<bg::ValueHolderPtr> &alloc_vec) {
  if (task_ret.size() != static_cast<size_t>(TaskProcKey::kNUM)) {
    GELOGE(ge::FAILED, "Task launch return num[%zu] is error", task_ret.size());
    return ge::FAILED;
  }
  if (graph_result == nullptr || graph_result->order_holders.empty()) {
    GELOGE(ge::FAILED, "Graph result IS NULL.");
    return ge::FAILED;
  }
  for (const auto &order_holder : graph_result->order_holders) {
    FE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(order_holder,
                                                           task_ret[static_cast<size_t>(TaskProcKey::H2D_COPY)]));
  }
  auto graph_res = const_cast<LowerResult*>(graph_result);
  graph_res->order_holders.clear();
  graph_res->order_holders.emplace_back(task_ret[static_cast<size_t>(TaskProcKey::TASK_LAUNCH)]);
  for (const auto &free_ptr : free_vec) {
    GELOGD("Available free nodes: %s after launch.", bg::ValueHolderUtils::GetNodeNameBarePtr(free_ptr));
    FE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(task_ret[static_cast<size_t>(TaskProcKey::TASK_LAUNCH)],
                                                           free_ptr));
  }
  for (const auto &alloc_ptr : alloc_vec) {
    GELOGD("Alloc node %s after launch.", bg::ValueHolderUtils::GetNodeNameBarePtr(alloc_ptr));
    alloc_ptr->ReleaseAfter(task_ret[static_cast<size_t>(TaskProcKey::TASK_LAUNCH)]);
  }
  return ge::SUCCESS;
}
std::vector<bg::ValueHolderPtr> RedirectLaunchArgs(const bg::ValueHolderPtr args_para) {
  if (args_para == nullptr) {
    GELOGE(ge::FAILED, "Node args para is null.");
    return {};
  }
  auto launch_arg = bg::ValueHolder::CreateDataOutput("RedirectLaunchArgs", {args_para},
                                                      static_cast<size_t>(AllocFFTSArgOutputs::kNum));
  return launch_arg;
}
} // namespace gert
