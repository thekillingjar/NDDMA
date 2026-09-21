/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/node_done_manager.h"
#include <chrono>
#include "framework/common/debug/ge_log.h"
#include "runtime/base.h"
#include "runtime/dev.h"
#include "exec_runtime/execution_runtime_utils.h"

namespace ge {
namespace hybrid {
namespace {
constexpr int32_t kDefaultWaitTimeoutInSec = 600;
constexpr int32_t kDefaultWaitTimeoutInSecForHeterogeneousRuntime = 1800;
}
std::string NodeDoneManager::Cond::ts_msg_;

bool NodeDoneManager::Cond::Await() {
  std::unique_lock<std::mutex> lk(cond_mu_);
  const auto timeout_in_sec = ExecutionRuntimeUtils::IsInHeterogeneousExecutor() ?
                              kDefaultWaitTimeoutInSecForHeterogeneousRuntime : kDefaultWaitTimeoutInSec;
  while (true) {
    if (!cv_.wait_for(lk,
                      std::chrono::seconds(timeout_in_sec),
                      [this]() { return is_released_ || is_cancelled_; })) {
      if (rtGetDevMsg(RT_GET_DEV_RUNNING_STREAM_SNAPSHOT_MSG, &DevMsgCallback) != RT_ERROR_NONE) {
        GELOGE(INTERNAL_ERROR, "[Call][RTS]Call rtGetDevMsg failed.");
        REPORT_INNER_ERR_MSG("E19999", "[Call][RTS]Call rtGetDevMsg failed.");
        return false;
      }
      if (subject_ts_msg_ != ts_msg_) {
        subject_ts_msg_ = ts_msg_;
        continue;
      } else {
        GELOGE(INTERNAL_ERROR, "[Invoke][wait_for]Wait timed out.");
        REPORT_INNER_ERR_MSG("E19999", "wait timed out[%d].", kDefaultWaitTimeoutInSec);
        return false;
      }
    }
    return is_released_;
  }

  return is_released_;
}

void NodeDoneManager::Cond::Reset() {
  const std::unique_lock<std::mutex> lk(cond_mu_);
  if ((!is_released_) && (!is_cancelled_)) {
    GELOGW("Called before done, released: %d, cancelled: %d",
           static_cast<int32_t>(is_released_), static_cast<int32_t>(is_cancelled_));
  }

  is_released_ = false;
  is_cancelled_ = false;
}

void NodeDoneManager::Cond::Release() {
  const std::unique_lock<std::mutex> lk(cond_mu_);
  is_released_ = true;
  cv_.notify_all();
}

void NodeDoneManager::Cond::Cancel() {
  const std::unique_lock<std::mutex> lk(cond_mu_);
  is_cancelled_ = true;
  cv_.notify_all();
}

bool NodeDoneManager::Cond::IsRelease() {
  const std::unique_lock<std::mutex> lk(cond_mu_);
  return is_released_;
}

NodeDoneManager::Cond *NodeDoneManager::GetSubject(const NodePtr &node) {
  const std::lock_guard<std::mutex> lk(mu_);
  if (destroyed_) {
    GELOGD("Already destroyed.");
    return nullptr;
  }

  const auto it = subjects_.find(node);
  if (it == subjects_.end()) {
    return &subjects_[node];
  }

  return &it->second;
}

void NodeDoneManager::Destroy() {
  GELOGD("Start to reset NodeDoneManager.");
  const std::lock_guard<std::mutex> lk(mu_);
  GELOGD("Cond size = %zu.", subjects_.size());
  for (auto &sub : subjects_) {
    if (!sub.second.IsRelease()) {
      sub.second.Cancel();
      GELOGD("[%s] Node canceled.", sub.first->GetName().c_str());
    }
  }

  subjects_.clear();
  destroyed_ = true;
  GELOGD("Done resetting NodeDoneManager successfully.");
}

void NodeDoneManager::NodeDone(const NodePtr &node) {
  const auto sub = GetSubject(node);
  if (sub != nullptr) {
    sub->Release();
    GELOGD("[%s] Node released.", node->GetName().c_str());
  }
}

bool NodeDoneManager::Await(const NodePtr &node) {
  const auto sub = GetSubject(node);
  if (sub == nullptr) {
    return false;
  }

  GELOGD("[%s] Await start. is_released = %s", node->GetName().c_str(), sub->IsRelease() ? "true" : "false");
  const bool ret = sub->Await();
  GELOGD("[%s] Await ended. is_released = %s", node->GetName().c_str(), sub->IsRelease() ? "true" : "false");
  return ret;
}

void NodeDoneManager::Reset(const NodePtr &node) {
  const auto sub = GetSubject(node);
  if (sub != nullptr) {
    sub->Reset();
    GELOGD("[%s] Node reset.", node->GetName().c_str());
  }
}

void NodeDoneManager::Reset() {
  GELOGD("NodeDoneManager reset all.");
  const std::lock_guard<std::mutex> lk(mu_);
  subjects_.clear();
  destroyed_ = false;
}
}  // namespace hybrid
}  // namespace ge
