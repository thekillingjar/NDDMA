/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow_session_manager.h"
#include <memory>
#include <utility>
#include "common/util/mem_utils.h"
#include "graph/manager/session_id_manager.h"

namespace ge {
namespace dflow {
using ge::Status;
using ge::SUCCESS;
using ge::FAILED;
using ge::GE_SESSION_MANAGER_NOT_INIT;
using ge::GE_SESSION_NOT_EXIST;
using ge::MEMALLOC_FAILED;

void DFlowSessionManager::Initialize() {
  if (init_flag_) {
    GELOGW("DFlowSession Manager has been initialized.");
    return;
  }
  init_flag_ = true;
}

void DFlowSessionManager::Finalize() {
  if (!init_flag_) {
    GELOGW("DFlowSession Manager has not been initialized.");
    return;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &item : session_manager_map_) {
    (void)item.second->Finalize();
  }
  session_manager_map_.clear();
  init_flag_ = false;
}

SessionPtr DFlowSessionManager::CreateSession(const std::map<std::string, std::string> &options,
                                              uint64_t &session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Create][Session]fail for DFlowSession manager is not initialized.");
    REPORT_INNER_ERR_MSG("E19999", "CreateSession fail for DFlowSession manager is not initialized.");
    return nullptr;
  }

  for (const auto &item : options) {
    GELOGI("option: %s, value: %s.", item.first.c_str(), item.second.c_str());
  }

  const std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t next_session_id = ge::SessionIdManager::GetNextSessionId();
  
  SessionPtr session_impl = ge::MakeShared<DFlowSessionImpl>(next_session_id, options);
  if (session_impl == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[Create][Session]fail for create session failed.");
    return nullptr;
  }
  if (session_impl->Initialize() != SUCCESS) {
    GELOGE(FAILED, "Dflow session initialize failed.");
    return nullptr;
  }
  (void)session_manager_map_.emplace(std::pair<uint64_t, SessionPtr>(next_session_id, session_impl));
  session_id = next_session_id;
  return session_impl;
}

Status DFlowSessionManager::DestroySession(uint64_t session_id) {
  if (!init_flag_) {
    GELOGW("[Destroy][Session]Session manager is not initialized, session_id:%lu.", session_id);
    return SUCCESS;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = session_manager_map_.find(session_id);
  if (it == session_manager_map_.end()) {
    return GE_SESSION_NOT_EXIST;
  }

  const SessionPtr &innerSession = it->second;
  const auto ret = innerSession->Finalize();
  if (ret != SUCCESS) {
    return ret;
  }
  (void)session_manager_map_.erase(session_id);
  return ret;
}

SessionPtr DFlowSessionManager::GetSession(uint64_t session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT,
           "[Get][Session]fail for DFlowSession manager is not initialized, session_id:%lu.", session_id);
    REPORT_INNER_ERR_MSG("E19999", "GetSession fail for DFlowSession manager is not initialized, session_id:%lu.",
                       session_id);
    return nullptr;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = session_manager_map_.find(session_id);
  if (it == session_manager_map_.end()) {
    GELOGE(GE_SESSION_NOT_EXIST, "[Find][InnerSession] fail for %lu does not exist", session_id);
    REPORT_INNER_ERR_MSG("E19999", "GetSession fail for InnerSession does not exist, session_id:%lu.", session_id);
    return nullptr;
  }
  return it->second;
}
} // namespace dflow
}  // namespace ge
