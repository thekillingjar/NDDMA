/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/tensorflow/scope/scope_pass_manager.h"
#include "parser/common/acl_graph_parser_util.h"
#include "common/util.h"
#include "base/err_msg.h"
#include "framework/common/debug/ge_log.h"
#include "register/scope/scope_graph_impl.h"
#include "register/scope/scope_pass_impl.h"

namespace ge {
std::shared_ptr<ScopeGraph> ScopePassManager::BuildScopeGraph(domi::tensorflow::GraphDef *graph_def) {
  GE_CHK_BOOL_EXEC(graph_def != nullptr, return nullptr, "graph_def is nullptr");
  scope_graph_ = ge::parser::MakeShared<ScopeGraph>();
  if (scope_graph_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New ScopeGraph failed");
    GELOGE(FAILED, "Scope graph make shared failed.");
    return nullptr;
  }
  Status ret = scope_graph_->Init();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "Scope graph init failed.");
    return nullptr;
  }

  auto &impl = scope_graph_->impl_;
  impl->BuildScopeGraph(graph_def);

  return scope_graph_;
}

Status ScopePassManager::AddPass(std::unique_ptr<ScopeBasePass> &pass) {
  GE_CHECK_NOTNULL(pass);

  graph_passes_.push_back(std::move(pass));

  return SUCCESS;
}

Status ScopePassManager::Run(std::shared_ptr<ScopeGraph> &graph) {
  GE_CHECK_NOTNULL(graph);
  bool not_changed = true;

  for (auto &pass : graph_passes_) {
    GE_CHECK_NOTNULL(pass);
    auto &impl = pass->impl_;
    if (impl == nullptr) {
      GELOGE(ge::MEMALLOC_FAILED, "ScopeBasePass is not properly initialized.");
      REPORT_INNER_ERR_MSG("E19999", "ScopeBasePass is not properly initialized");
      continue;
    }

    Status status = impl->Run(graph);
    if (status == SUCCESS) {
      GELOGI("Run scope pass:%s success.", pass->PassName().c_str());
      not_changed = false;
    } else if (status != domi::SCOPE_NOT_CHANGED) {
      // exception
      REPORT_INNER_ERR_MSG("E19999", "Run scope fusion pass [%s] failed.", pass->PassName().c_str());
      GELOGE(FAILED, "Pass Run failed, pass name:%s", pass->PassName().c_str());
      return status;
    }
  }

  return not_changed ? domi::SCOPE_NOT_CHANGED : SUCCESS;
}
}  // namespace ge
