/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fusion_pass_executor.h"
#include "pass_registry.h"
#include "common/debug/ge_log.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/checker.h"
#include "common/util/trace_manager/trace_manager.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "register/custom_pass_helper.h"
#include "register/custom_pass_context_impl.h"
#include "graph/fusion/fusion_utils.h"
#include "register/pass_option_utils.h"
namespace ge {
namespace fusion {
namespace {
const size_t kMaxRepassTimes = 10U;
const std::string kPassSwitchAll = "ALL";
bool IsPassEnable(const std::map<std::string, bool> &pass_name_2_switches, const std::string &pass_name) {
  bool is_enable_by_option = false;
  if (PassOptionUtils::CheckIsPassEnabledByOption(pass_name, is_enable_by_option) == SUCCESS) {
    return is_enable_by_option;
  }
  const auto iter = pass_name_2_switches.find(pass_name);
  if (iter != pass_name_2_switches.cend()) {
    return iter->second;
  }

  auto all_iter = pass_name_2_switches.find(kPassSwitchAll);
  if (all_iter != pass_name_2_switches.end()) {
    return all_iter->second;
  }
  return true;
}
Status MergeFinalStatus(Status final_status, Status cur_pass_status) {
  if (final_status != NOT_CHANGED && final_status != SUCCESS) {
    return final_status;
  }
  return cur_pass_status == NOT_CHANGED ? final_status : cur_pass_status;
}
} // namespace

Status FusionPassExecutor::RunPasses(const ComputeGraphPtr &compute_graph, CustomPassStage stage) {
  GE_ASSERT_SUCCESS(InitPassesIfNeed(stage));
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  FusionUtils::ResetCycleDetectors();
  GE_MAKE_GUARD(reset_cycle_detector, [] { FusionUtils::ResetCycleDetectors(); });
  CustomPassContext context;
  // todo if changed, run it next time
  Status final_status = NOT_CHANGED;
  for (auto &pass_pair : names_to_fusion_passes_) {
    const auto &pass_name = pass_pair.first;
    const auto &pass = pass_pair.second;
    GELOGD("[Run][FusionPass] %s in stage %s", pass_name.c_str(), CustomPassStageToString(stage).c_str());
    GE_CHECK_NOTNULL(pass);
    TraceOwnerGuard guard("Fusion", pass_name, compute_graph->GetName());
    GE_TRACE_START(FusionPassRun);
    context.SetPassName(pass_name.c_str());
    Status status = pass->Run(graph, context);
    final_status = MergeFinalStatus(final_status, status);
    if ((final_status != SUCCESS) && (final_status != NOT_CHANGED)) {
      GELOGE(final_status, "[%s][Run] failed on graph %s", pass_name.c_str(), compute_graph->GetName().c_str());
      return final_status;
    }

    for (const auto &sub_compute_graph :compute_graph->GetAllSubgraphs()) {
      GE_CHECK_NOTNULL(sub_compute_graph);
      std::string subgraph_pass_name = pass_name + "::" + compute_graph->GetName();
      GE_TRACE_START(PassRunSubgraph);
      auto subgraph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(sub_compute_graph);
      TraceOwnerGuard sub_guard("GE_SUB", subgraph_pass_name, compute_graph->GetName());
      status = pass->Run(subgraph, context);
      GE_COMPILE_TRACE_TIMESTAMP_END(PassRunSubgraph, subgraph_pass_name.c_str());
      if ((status != SUCCESS) && (status != NOT_CHANGED)) {
        GELOGE(status, "[%s][Run] failed on subgraph %s", pass_name.c_str(), sub_compute_graph->GetName().c_str());
        return status;
      }
    }
    GE_COMPILE_TRACE_TIMESTAMP_END(FusionPassRun, pass_name.c_str());
  }
  return SUCCESS;
}

FusionPassExecutor::~FusionPassExecutor() {
  for (auto &pass_pair : names_to_fusion_passes_) {
    auto &pass = pass_pair.second;
    GE_DELETE_NEW_SINGLE(pass);
  }
}

Status FusionPassExecutor::RunPassesWithLegacyCustom(const ComputeGraphPtr &compute_graph, CustomPassStage stage) {
  CustomPassContext context;
  auto graph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  GE_ASSERT_SUCCESS(CustomPassHelper::Instance().Run(graph, context, stage),
                    "Run custom pass for graph [%s] failed.", compute_graph->GetName().c_str());
  GE_ASSERT_SUCCESS(RunPasses(compute_graph, stage));
  return SUCCESS;
}

Status FusionPassExecutor::InitPassesIfNeed(CustomPassStage stage) {
  if (!names_to_fusion_passes_.empty()) {
    return SUCCESS;
  }
  if (pass_name_to_switches_.empty()) {
    pass_name_to_switches_ = FusionUtils::ParseFusionSwitch();
  }
  auto pass_creators = PassRegistry::GetInstance().GetFusionPassRegDataByStage(stage);
  for (const auto &pass_reg : pass_creators) {
    auto pass_creator = pass_reg.GetCreatePassFn();
    GE_ASSERT_NOTNULL(pass_creator);
    const std::string pass_name = pass_reg.GetPassName().GetString();
    if (!IsPassEnable(pass_name_to_switches_, pass_name)) {
      GELOGI("[FusionPass][SKIP] Pass [%s] is disabled by fusion switch config file, Option[%s][%s].",
             pass_reg.ToString().GetString(), FUSION_SWITCH_FILE.c_str(),
             FusionUtils::GetFusionSwitchFileFromOption().c_str());
      continue;
    }
    names_to_fusion_passes_.emplace_back(pass_name, pass_creator());
    GELOGD("[FusionPass][ADD] %s", pass_reg.ToString().GetString());
  }
  return SUCCESS;
}
} // namespace fusion
}  // namespace ge