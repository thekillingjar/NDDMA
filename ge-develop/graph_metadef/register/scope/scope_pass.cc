/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <stack>
#include "register/scope/scope_pass_impl.h"
#include "register/scope/scope_graph_impl.h"
#include "register/scope/scope_pattern_impl.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace ge {
ScopesResult::ScopesResult() {
  impl_ = ge::ComGraphMakeUnique<ScopesResultImpl>();
}

ScopesResult::ScopesResult(ScopesResult const &result) {
  impl_ = ge::ComGraphMakeUnique<ScopesResultImpl>();
  if ((impl_ == nullptr) || (result.impl_ == nullptr)) {
    GELOGE(ge::MEMALLOC_FAILED, "ScopesResult is not properly initialized.");
    return;
  }
  const std::vector<Scope *> &scopes = result.impl_->GetScopes();
  const std::vector<ge::OperatorPtr> &nodes = result.impl_->GetNodes();
  impl_->SetScopes(scopes);
  impl_->SetNodes(nodes);
}
ScopesResult &ScopesResult::operator=(ScopesResult const &result) {
  if (&result == this) {
    return *this;
  }
  if ((impl_ == nullptr) || (result.impl_ == nullptr)) {
    GELOGE(ge::MEMALLOC_FAILED, "ScopesResult is not properly initialized.");
    return *this;
  }
  const std::vector<Scope *> &scopes = result.impl_->GetScopes();
  const std::vector<ge::OperatorPtr> &nodes = result.impl_->GetNodes();
  impl_->SetScopes(scopes);
  impl_->SetNodes(nodes);
  return *this;
}

ScopesResult::~ScopesResult() = default;

void ScopesResult::SetScopes(std::vector<Scope *> &scopes) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetScopes(), ScopesResult is not properly initialized.");
    return;
  }

  impl_->SetScopes(scopes);
}

void ScopesResult::SetNodes(std::vector<ge::OperatorPtr> &nodes) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetNodes(), ScopesResult is not properly initialized.");
    return;
  }

  impl_->SetNodes(nodes);
}

ScopeBasePass::ScopeBasePassImpl::~ScopeBasePassImpl() {
  for (auto &scope_patterns : patterns_) {
    for (auto &batch_patterns : scope_patterns) {
      for (auto &pattern : batch_patterns) {
        if (pattern != nullptr) {
          delete pattern;
          pattern = nullptr;
        }
      }
    }
  }
}

Status ScopeBasePass::ScopeBasePassImpl::AddFusionScopesResultToScopeGraph(
    const std::shared_ptr<ScopeGraph> &scope_graph, std::vector<ScopesResult> &scope_results) const {
  for (auto &rlt : scope_results) {
    std::unique_ptr<FusionScopesResult> fusion_rlt = ComGraphMakeUnique<FusionScopesResult>();
    if (fusion_rlt == nullptr) {
      GELOGE(FAILED, "Alloc fusion_rlt failed.");
      return FAILED;
    }
    if (fusion_rlt->Init() != SUCCESS) {
      GELOGE(FAILED, "Init fusion_rlt failed.");
      return FAILED;
    }
    auto &impl_fusion_rlt = fusion_rlt->impl_;
    auto &impl_scope_rlt = rlt.impl_;
    if (impl_scope_rlt == nullptr) {
      GELOGE(ge::MEMALLOC_FAILED, "ScopesResult is not properly initialized.");
      continue;
    }

    impl_fusion_rlt->AddNodes(impl_scope_rlt->GetNodes());
    impl_fusion_rlt->AddScopes(impl_scope_rlt->GetScopes());
    parent_->GenerateFusionResult(impl_scope_rlt->GetScopes(), fusion_rlt.get());
    if (impl_fusion_rlt->Type() == kScopeInvalidType) {
      GELOGE(FAILED, "Failed to set inner node for fusion op %s.", impl_fusion_rlt->Type().c_str());
      return FAILED;
    }
    auto &impl_scope_graph = scope_graph->impl_;
    impl_scope_graph->AddFusionScopesResult(fusion_rlt.release());
  }

  return SUCCESS;
}

Status ScopeBasePass::ScopeBasePassImpl::Run(std::shared_ptr<ScopeGraph> &scope_graph) {
  GE_CHECK_NOTNULL(scope_graph);
  const ScopeTree *const scope_tree = scope_graph->GetScopeTree();
  GE_CHECK_NOTNULL(scope_tree);
  GE_CHECK_NOTNULL(parent_);
  patterns_ = parent_->DefinePatterns();
  std::vector<Scope *> results;
  if (!MatchAllBatches(scope_tree, results)) {
    GELOGI("[scope_fusion] Scope pass %s's patterns is not matched and ignored.", parent_->PassName().c_str());
    return domi::SCOPE_NOT_CHANGED;
  }
  GELOGI("[scope_fusion] Scope pass %s's patterns is matched.", parent_->PassName().c_str());

  std::vector<ScopesResult> scope_results;
  Status ret = parent_->LastMatchScopesAndOPs(scope_graph, scope_results);
  if (ret != SUCCESS) {
    for (auto &result : results) {
      GE_CHECK_NOTNULL(result);
      auto &impl_scope = result->impl_;
      impl_scope->ClearTypeAndSubType();
    }
    GELOGW("[ScopeFusion][RunPass] Scope pass %s's patterns is ignored, because LastMatchScopesAndOPs failed.",
           parent_->PassName().c_str());
    return domi::SCOPE_NOT_CHANGED;
  }

  if (!results.empty()) {
    ret = AddFusionScopesResultToScopeGraph(scope_graph, scope_results);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Scope pass %s add fusion scopes result to scope graph failed.", parent_->PassName().c_str());
      return domi::SCOPE_NOT_CHANGED;
    }
  } else {
    GELOGI("[scope_fusion] Scope pass %s not match any scope.", parent_->PassName().c_str());
  }

  ret = PrintFusionScopeInfo(scope_graph);
  if (ret != SUCCESS) {
    GELOGI("[scope_fusion] Can not print scope pass %s fusion info.", parent_->PassName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

bool ScopeBasePass::ScopeBasePassImpl::MatchAllBatches(const ScopeTree *scope_tree, std::vector<Scope *> &results) {
  if (scope_tree == nullptr) {
    GELOGE(PARAM_INVALID, "Input param [scope_tree] is nullptr.");
    return false;
  }

  for (auto &scope_patterns : patterns_) {
    std::vector<Scope *> tmp_results;
    std::vector<Scope *> last_results;
    uint32_t batch_num = 0U;
    for (auto &batch_patterns : scope_patterns) {
      ++batch_num;
      std::vector<Scope *> one_results;
      const bool is_matched = MatchOneBatch(scope_tree, batch_patterns, one_results);
      if (!is_matched) {
        break;
      }
      if (batch_num == scope_patterns.size()) {
        (void)last_results.insert(last_results.cend(), one_results.cbegin(), one_results.cend());
      } else {
        (void)tmp_results.insert(tmp_results.cend(), one_results.cbegin(), one_results.cend());
      }
    }
    for (auto &tmp : tmp_results) {
      bool rollback = true;
      for (auto &result : last_results) {
        AscendString result_name;
        AscendString tmp_name;
        (void) result->Name(result_name);
        (void) tmp->Name(tmp_name);
        if ((result_name.GetLength() <= tmp_name.GetLength()) && (tmp_name.Find(result_name) == 0U)) {
          rollback = false;
          break;
        }
      }
      if (rollback) {
        auto &impl = tmp->impl_;
        impl->SetSubType("");
      }
    }
    (void)results.insert(results.cend(), last_results.cbegin(), last_results.cend());
  }

  return !(results.empty());
}

bool ScopeBasePass::ScopeBasePassImpl::MatchOneBatch(const ScopeTree *const scope_tree,
                                                     const std::vector<ScopePattern *> &patternlist,
                                                     std::vector<Scope *> &results) const {
  if (scope_tree == nullptr) {
    GELOGE(PARAM_INVALID, "Input param [scope_tree] is nullptr");
    return false;
  }

  int32_t find = 0;
  auto &impl_scope_tree = scope_tree->impl_;
  const Scope *const root = impl_scope_tree->Root();
  if (root != nullptr) {
    auto &impl_scope = root->impl_;
    const std::unordered_map<std::string, Scope *> &sub_scopes = impl_scope->GetSubScopes();
    for (auto &pattern : patternlist) {
      for (auto &scope : sub_scopes) {
        if (MatchOneScope(pattern, scope.second, results)) {
          ++find;
        }
      }
    }
  }

  return (find > 0) ? true : false;
}

bool ScopeBasePass::ScopeBasePassImpl::MatchOneScope(const ScopePattern *pattern, Scope *scope,
                                                     std::vector<Scope *> &results) const {
  if ((pattern == nullptr) || (scope == nullptr)) {
    GELOGE(PARAM_INVALID, "Input param is nullptr");
    return false;
  }
  auto &impl_scope_pattern = pattern->impl_;
  if (impl_scope_pattern == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "ScopePattern is not properly initialized.");
    return false;
  }
  if (impl_scope_pattern->Match(scope)) {
    auto &scope_impl = scope->impl_;
    scope_impl->SetSubType(impl_scope_pattern->SubType());
    results.push_back(scope);
    return true;
  }
  int32_t find = 0;
  std::stack<Scope *> scopes;
  scopes.push(scope);
  while (!scopes.empty()) {
    const Scope *const current_scope = scopes.top();
    scopes.pop();
    auto &current_scope_impl = current_scope->impl_;
    const std::unordered_map<std::string, Scope *> &sub_scopes = current_scope_impl->GetSubScopes();
    for (auto &sub_scope : sub_scopes) {
      if (impl_scope_pattern->Match(sub_scope.second)) {
        auto &sub_scope_impl = sub_scope.second->impl_;
        sub_scope_impl->SetSubType(impl_scope_pattern->SubType());
        results.push_back(sub_scope.second);
        ++find;
      } else {
        scopes.push(sub_scope.second);
      }
    }
  }
  return (find > 0) ? true : false;
}

Status ScopeBasePass::ScopeBasePassImpl::PrintFusionScopeInfo(std::shared_ptr<ScopeGraph> &scope_graph) const {
  if (scope_graph == nullptr) {
    GELOGE(PARAM_INVALID, "Input param scope_graph is nullptr.");
    return PARAM_INVALID;
  }
  auto &impl_scope_graph = scope_graph->impl_;
  const std::unordered_map<std::string, FusionScopesResult *> &final_results = impl_scope_graph->FusionScopesResults();
  for (auto &result : final_results) {
    if (result.second == nullptr) {
       GELOGE(PARAM_INVALID, "Fusion scope is nullptr.");
       return PARAM_INVALID;
    }
    AscendString name;
    (void) result.second->Name(name);
    GELOGI("FusionScope:%s", name.GetString());
    auto &impl = result.second->impl_;
    const std::map<std::string, std::vector<int32_t>> &inputs = impl->GetInputs();
    for (auto &input : inputs) {
      const std::vector<int32_t> indexs = input.second;
      for (const int32_t index : indexs) {
        GELOGI("FusionScope input node:%s,%d", input.first.c_str(), index);
      }
    }

    const std::map<std::string, std::vector<int32_t>> &outputs = impl->GetOutputs();
    for (auto &output : outputs) {
      const std::vector<int32_t> indexs = output.second;
      for (const int32_t index : indexs) {
        GELOGI("FusionScope output node:%s,%d", output.first.c_str(), index);
      }
    }

    for (auto &scope : impl->Scopes()) {
      if (scope == nullptr) {
        GELOGE(PARAM_INVALID, "Scope in fusion scope is nullptr.");
        return PARAM_INVALID;
      }
      AscendString scope_name;
      (void) scope->Name(scope_name);
      GELOGI("FusionScope GetScope:%s", scope_name.GetString());
    }

    for (auto &node : result.second->Nodes()) {
      if (node == nullptr) {
        GELOGE(PARAM_INVALID, "Node in scope is nullptr.");
        return PARAM_INVALID;
      }
      AscendString node_name;
      (void) node->GetName(node_name);
      GELOGI("FusionScope Node:%s", node_name.GetString());
    }
  }
  return SUCCESS;
}

ScopeBasePass::ScopeBasePass() {
  impl_ = ge::ComGraphMakeUnique<ScopeBasePassImpl>(this);
}

ScopeBasePass::~ScopeBasePass() = default;
}  // namespace ge
