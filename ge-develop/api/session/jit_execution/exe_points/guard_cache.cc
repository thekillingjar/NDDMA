/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <memory>

#include "framework/common/debug/ge_log.h"
#include "guard_cache.h"
#include "execution_point.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/checker.h"
#include "base/err_msg.h"


using namespace ge;

GuardedExecutionPoint *GuardCheckCache::FindGuardedExecutionPoint(const std::vector<gert::Tensor> &input_tensor) {
  int64_t ep_id = -1;
  if (owner_point_) {
    ep_id = owner_point_->GetId();
  }
  for (auto &item : cache_models_) {
    auto result = item->Match(input_tensor);
    if (result) {
      GELOGI("Get EP[%ld] hint GEP(%u) and priority is (%u)", ep_id, item->GetCompiledGraphId(),
             item->GetPriority());
      item->SetPriority(item->GetPriority() + 1);
      return item.get();
    }
  }
  GELOGI("There is no hint GEP in EP[%ld] cache size(%lu)", ep_id, cache_models_.size());
  return nullptr;
}

uint32_t GuardCheckCache::GetSavedCacheNum() const {
  return cache_models_.size();
}

GuardedExecutionPoint *GuardCheckCache::FindOrCreateGuarded(const std::vector<gert::Tensor> &inputs) {
  GuardedExecutionPoint *gep = FindGuardedExecutionPoint(inputs);
  if (gep) {
    return gep;
  }

  if (owner_point_) {
    gep = FindGuardedExecutionPoint(inputs);
    if (gep) {
      return gep;
    }
    // 第一个GEP依然是Guard Miss，走下面及CompileAndLoad流程进行编译
  }
  REPORT_INNER_ERR_MSG("W18888", "There is no hint GEP, cache size(%lu). Guard miss reason in info log", cache_models_.size());
  gep = new GuardedExecutionPoint(owner_point_);
  GE_ASSERT_SUCCESS(AddCompiledCompiledGraph(gep));

  return gep;
}

Status GuardCheckCache::AddCompiledCompiledGraph(GuardedExecutionPoint* gep) {
  if (!gep) {
    return FAILED;
  }

  // 拷贝EP中的原图，用于gep->GetGraph()被CompileAndLoad使用
  GE_ASSERT_SUCCESS(gep->CopySlicedGraph());

  if (max_cache_count_ <= cache_models_.size()) {
    auto &lastItem = cache_models_.back();
    lastItem->RemoveItem();
    cache_models_.pop_back();
  }

  cache_models_.emplace_back(gep);

  std::sort(cache_models_.begin(), cache_models_.end(),
            [](std::unique_ptr<ge::GuardedExecutionPoint> &a, std::unique_ptr<ge::GuardedExecutionPoint> &b) {
              return a->GetPriority() > b->GetPriority();
            });

  return SUCCESS;
}

Status GuardCheckCache::RemoveCompiledGraph() {
  for (auto& item : cache_models_) {
    auto status = item->RemoveItem();
    if (status != ge::SUCCESS) {
      return status;
    }
  }
  cache_models_.clear();
  return ge::SUCCESS;
}
