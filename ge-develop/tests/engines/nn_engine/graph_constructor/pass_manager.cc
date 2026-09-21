/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pass_manager.h"
#include "common/configuration.h"
#include "common/fe_error_code.h"
#include "common/fe_log.h"
#include "common/fe_report_error.h"
#include "graph_optimizer/fusion_common/graph_node_map_util.h"
#include "common/util/trace_manager/trace_manager.h"
namespace fe {
const vector<GraphPass *> &PassManager::GraphPasses() { return graph_passes_; }

Status PassManager::AddPass(const std::string &pass_name, const std::string &engine_name, GraphPass *pass,
                            const std::string fusion_type) {
  FE_CHECK(pass == nullptr, REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][AddPass] pass is null, AddPass failed"),
           return PARAM_INVALID);
  graph_passes_.push_back(pass);

  return SUCCESS;
}

Status PassManager::Run(ge::ComputeGraph &graph) { return Run(graph, graph_passes_); }

Status PassManager::Run(ge::ComputeGraph &graph, vector<GraphPass *> &passes) {
  bool not_changed = true;

  for (auto pass : passes) {
    FE_CHECK(pass == nullptr, REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][RUN] pass is null, Run pass failed"),
             return PARAM_INVALID);
    ge::TraceOwnerGuard guard(FE_MODULE_NAME, pass->GetName(), graph.GetName());
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

Status PassManager::Run(ge::ComputeGraph &graph, OpsKernelInfoStorePtr ops_kernel_info_store_ptr) {
  return Run(graph, graph_passes_, ops_kernel_info_store_ptr);
}

Status PassManager::Run(ge::ComputeGraph &graph, vector<GraphPass *> &passes,
                        OpsKernelInfoStorePtr ops_kernel_info_store_ptr) {
  bool not_changed = true;

  for (auto &pass : passes) {
    FE_CHECK(pass == nullptr, REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][RUN] pass is null, Run pass failed"),
             return PARAM_INVALID);

    PatternFusionBasePass *convert_pass = dynamic_cast<PatternFusionBasePass *>(pass);
    FE_CHECK(convert_pass == nullptr,
             REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][RUN] Pass[%s]: fail to dynamic_cast GraphPass to \
             PatternFusionBasePass.", pass->GetName().c_str()),
             return FAILED);
    ge::TraceOwnerGuard guard(FE_MODULE_NAME, pass->GetName(), graph.GetName());
    Status status = convert_pass->Run(graph, ops_kernel_info_store_ptr);
    // if pass takes effect, set not_changed to be false
    if (status == SUCCESS) {
      not_changed = false;
    } else if (status != NOT_CHANGED) {
      ErrorMessageDetail error_msg(EM_RUN_PASS_FAILED, {pass->GetName(), "built-in-ai-core-graph-pass"});
      ReportErrorMessage(error_msg);
      // error situation
      REPORT_FE_ERROR("[GraphOpt][FirstRoundFusion][RUN] Pass Run failed, status:%d", status);
      return status;
    }
  }

  return not_changed ? NOT_CHANGED : SUCCESS;
}

PassManager::PassManager() {}

PassManager::~PassManager() {
  for (auto pass : graph_passes_) {
    if (pass != nullptr) {
      delete pass;
      pass = nullptr;
    }
  }
}
}  // namespace fe
