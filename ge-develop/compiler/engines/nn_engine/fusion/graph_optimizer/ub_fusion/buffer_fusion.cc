/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include <memory>
#include "common/configuration.h"
#include "common/op_info_common.h"
#include "graph_optimizer/ub_fusion/buffer_fusion_optimizer.h"
#include "graph_optimizer/ub_fusion/auto_buffer_fusion_pass_runner.h"

namespace fe {
/*
 * @brief: match defined fusion pattern from graph and assign scope id to fusion
 * op
 */
Status BufferFusion::MatchFusionPatternFromGraph(ge::ComputeGraph &graph) {
  // ub fusion, te fusion && cce aicore fusion
  if (MatchFusionPattern(graph) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][MtcFusPtn] Failed to perform UB fusion.");
    return FAILED;
  }

  return SUCCESS;
}

/*
 * @brief: create fusion graph with scope_id create by
 * MatchFusionPatternFromGraph,
 *        i.e. nodes have same scope_id will be fused into one fusion op,
 *        the topo of graph will be changed.
 */
Status BufferFusion::BuildFusionGraph(ge::ComputeGraph &graph) {
  // merge fusion node
  if (Configuration::Instance(engine_name_).EnableL1Fusion()) {
    FE_CHECK_NOTNULL(l1_fusion_graph_merge_ptr_);
    if (l1_fusion_graph_merge_ptr_->MergeFusionGraph(graph) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][PostProcess][BuildFusGraph] Failed to merge the fusion graph.");
      return FAILED;
    }
  }

  FE_CHECK_NOTNULL(ub_fusion_graph_merge_ptr_);
  if (ub_fusion_graph_merge_ptr_->MergeFusionGraph(graph) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][BuildFusGraph] Failed to merge the fusion graph.");
    return FAILED;
  }
  return SUCCESS;
}

Status BufferFusion::MatchFusionPattern(ge::ComputeGraph &graph) {
  if (cycle_detector_ == nullptr) {
    FE_MAKE_SHARED(cycle_detector_ = std::make_shared<FusionCycleDetector>(), return FAILED);
    FE_CHECK_NOTNULL(cycle_detector_);
    cycle_detector_->Initialize(graph);
  }

  Status ret = RunBuiltInFusion(graph);
  if (ret != SUCCESS) {
    return ret;
  }
  return SUCCESS;
}

Status BufferFusion::RunBuiltInFusion(ge::ComputeGraph &graph) {
  BufferFusionPassType pass_type = (engine_name_ == fe::AI_CORE_NAME) ? BUILT_IN_AI_CORE_BUFFER_FUSION_PASS
                                                                      : BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS;
  if (RunRegisterBufferFusionPass(graph, pass_type) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

BaseBufferFusionPassRunnerPtr BufferFusion::MakePassRunnerPtr(const BufferFusionInfo &sorted_buffer_fusion_info,
                                                const BufferFusionOptimizerPtr buffer_fusion_optimizer) const {
  BaseBufferFusionPassRunnerPtr buffer_fusion_pass_runner_ptr = nullptr;
  if (sorted_buffer_fusion_info.is_auto_fusion) {
    // auto fusion framework
    FE_MAKE_SHARED(buffer_fusion_pass_runner_ptr = std::make_shared<AutoBufferFusionPassRunner>(
        sorted_buffer_fusion_info.name, sorted_buffer_fusion_info.buffer_fusion_pass_create_fn,
        cycle_detector_, op_store_adapter_ptr_),
    return nullptr);
  } else {
    // normal fusion framework
    FE_MAKE_SHARED(buffer_fusion_pass_runner_ptr = std::make_shared<BufferFusionPassRunner>(
        sorted_buffer_fusion_info.name, sorted_buffer_fusion_info.buffer_fusion_pass_create_fn, cycle_detector_,
        op_store_adapter_ptr_, sorted_buffer_fusion_info.is_fusion_check, buffer_fusion_optimizer),
    return nullptr);
  }
  return buffer_fusion_pass_runner_ptr;
}

Status BufferFusion::RunRegisterBufferFusionPass(ge::ComputeGraph &graph, BufferFusionPassType pass_type) {
  string pass_type_str = GetBufferFusionPassTypeString(pass_type);
  FE_LOGD("Graph[%s] PassType[%s] start to run reg buffer fusion pass.", graph.GetName().c_str(), pass_type_str.c_str());
  FE_CHECK(fusion_priority_mgr_ptr_ == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][PostProcess][RunRegBufFus] The fusion_priority_mgr_ptr_ is null."),
           return FAILED);
  const std::vector<BufferFusionInfo> &sorted_buffer_fusion_vec =
          fusion_priority_mgr_ptr_->GetSortedBufferFusionList(graph);
  if (sorted_buffer_fusion_vec.empty()) {
    FE_LOGD("No fusion pass got read, BufferFusionPassType:[%u].", pass_type);
    return SUCCESS;
  }

  BufferFusionOptimizerPtr buffer_fusion_optimizer = nullptr;
  FE_MAKE_SHARED(buffer_fusion_optimizer = std::make_shared<BufferFusionOptimizer>(), return FAILED);
  buffer_fusion_optimizer->Initialize(graph);
  for (const auto &sorted_buffer_fusion_info : sorted_buffer_fusion_vec) {
    FE_LOGD("Start buffer fusion: %s, Priority: %d.", sorted_buffer_fusion_info.name.c_str(),
            FusionPriorityManager::GetRealPriority(sorted_buffer_fusion_info.priority));
    int32_t priority = FusionPriorityManager::GetRealPriority(sorted_buffer_fusion_info.priority);
    FE_LOGD_IF(priority < CUSTOM_PASS_PRIORITY_MIN,
               "Start to run buffer fusion, pass name:%s, pass type:%s, configured priority:%d, engine:%s.",
               sorted_buffer_fusion_info.name.c_str(), pass_type_str.c_str(), priority, engine_name_.c_str());
    FE_LOGD_IF(priority >= CUSTOM_PASS_PRIORITY_MIN,
               "Start to run buffer fusion, pass name:%s, pass type:%s, default priority:%d, engine:%s.",
               sorted_buffer_fusion_info.name.c_str(), pass_type_str.c_str(), priority, engine_name_.c_str());
    auto buffer_fusion_pass_runner_ptr = MakePassRunnerPtr(sorted_buffer_fusion_info, buffer_fusion_optimizer);
    FE_CHECK_NOTNULL(buffer_fusion_pass_runner_ptr);
    Status ret = buffer_fusion_pass_runner_ptr->Run(graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][UBMatch] Buffer fusion pass execution failed, result: %u, graph: %s, pass: %s, type: %s",
                      ret, graph.GetName().c_str(), sorted_buffer_fusion_info.name.c_str(), pass_type_str.c_str());
      return ret;
    }
    FE_LOGI("Run buffer fusion pass successfully, pass name:%s.", sorted_buffer_fusion_info.name.c_str());
  }
  FE_LOGD("GraphName[%s] PassType[%s]: register buffer fusion pass has ended.", graph.GetName().c_str(),
          pass_type_str.c_str());
  return SUCCESS;
}
}  // namespace fe
