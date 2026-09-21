/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_pass_manager.h"
#include "common/configuration.h"
#include "common/fe_error_code.h"
#include "common/fe_log.h"
#include "common/fe_report_error.h"
#include "graph_optimizer/fusion_common/graph_node_map_util.h"
#include "common/util/trace_manager/trace_manager.h"

namespace fe {
Status UbPassManager::AddPass(const std::string &pass_name, const std::string &engine_name, BaseBufferFusionPassRunner *pass,
                            const std::string fusion_type) {
  FE_CHECK(pass == nullptr, REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][AddPass] pass is null, AddPass failed"),
           return PARAM_INVALID);
  graph_passes_.push_back(pass);

  return SUCCESS;
}

Status UbPassManager::Run(const ge::ComputeGraph &graph) { return Run(graph, graph_passes_); }

Status UbPassManager::Run(const ge::ComputeGraph &graph, vector<BaseBufferFusionPassRunner *> &passes) {
  bool not_changed = true;
  for (auto pass : passes) {
    FE_CHECK(pass == nullptr, REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][RUN] pass is null, Run pass failed"),
             return PARAM_INVALID);
    ge::TraceOwnerGuard guard(FE_MODULE_NAME, graph.GetName(), graph.GetName());
    Status status = pass->Run(graph);
    // if pass takes effect, set not_changed to be false
    if (status == SUCCESS) {
      not_changed = false;
    } else if (status != NOT_CHANGED) {
      // error situation
      REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][RUN] Pass Run failed, status:%d", status);
      return status;
    }
  }

  return not_changed ? NOT_CHANGED : SUCCESS;
}

UbPassManager::UbPassManager() {}

UbPassManager::~UbPassManager() {
  for (auto pass : graph_passes_) {
    if (pass != nullptr) {
      delete pass;
      pass = nullptr;
    }
  }
}
}  // namespace fe
