/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "resource_mgr.h"
#include "framework/common/debug/ge_log.h"
#include "base/err_msg.h"

namespace gert {
ResourceMgr::~ResourceMgr() {
    const std::lock_guard<std::mutex> lock(mu_);
    handle_map_.clear();
    step_handle_.clear();
}

ge::graphStatus ResourceMgr::Create(const uint64_t handle, TensorSeqPtr resource) {
    GELOGD("Create handle: [%llu]", handle);
    const std::lock_guard<std::mutex> lock(mu_);
    if (handle_map_.count(handle) != 0) {
        GELOGE(ge::PARAM_INVALID, "handle [%llu] has already exist.", handle);
        REPORT_INNER_ERR_MSG("E39999", "handle has already exist.");
        return ge::PARAM_INVALID;
    } else {
        handle_map_[handle] = resource;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResourceMgr::Lookup(const uint64_t handle, TensorSeqPtr* resource) {
    GELOGD("Lookup handle: [%llu]", handle);
    const std::lock_guard<std::mutex> lock(mu_);
    const auto iter = handle_map_.find(handle);
    if (iter == handle_map_.end()) {
        GELOGE(ge::PARAM_INVALID, "handle [%llu] does not exist.", handle);
        REPORT_INNER_ERR_MSG("E39999", "handle does not exist.");
        return ge::PARAM_INVALID;
    }
    *resource = iter->second;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResourceMgr::ClearSpecialStepResource(const uint64_t handle) {
    GELOGD("Clear handle: [%llu]", handle);
    const std::lock_guard<std::mutex> lock(mu_);
    for (auto &h : step_handle_) {
        if (h == handle) {
            const auto iter = handle_map_.find(h);
            if (iter == handle_map_.end()) {
                GELOGE(ge::PARAM_INVALID, "handle [%llu] does not exist.", h);
                REPORT_INNER_ERR_MSG("E39999", "handle does not exist.");
                return ge::PARAM_INVALID;
            }
            step_handle_.erase(h);
            handle_map_.erase(iter);
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::PARAM_INVALID;
}

ge::graphStatus ResourceMgr::ClearStepResource() {
    const std::lock_guard<std::mutex> lock(mu_);
    for (auto it = step_handle_.begin(); it != step_handle_.end();) {
        const auto iter = handle_map_.find(*it);
        if (iter == handle_map_.end()) {
            step_handle_.erase(it++);
            GELOGE(ge::PARAM_INVALID, "handle [%llu] does not exist.", *it);
            REPORT_INNER_ERR_MSG("E39999", "handle does not exist.");
            continue;
        }
        step_handle_.erase(it++);
        handle_map_.erase(iter);
    }
    return ge::GRAPH_SUCCESS;
}

void ResourceMgr::ClearAllResource() {
    const std::lock_guard<std::mutex> lock(mu_);
    handle_map_.clear();
    step_handle_.clear();
    return;
}

SessionMgr* SessionMgr::GetInstance() {
    static SessionMgr inst;
    return &inst;
}

SessionMgr::~SessionMgr() {
    const std::lock_guard<std::mutex> lock(session_mutex_);
    session_map_.clear();
}

void SessionMgr::GetOrCreateSession(const uint64_t session_id, SessionPtr& session) {
    GELOGD("GetOrCreate session_id: [%llu]", session_id);
    const std::lock_guard<std::mutex> lock(session_mutex_);
    const auto iter = session_map_.find(session_id);
    if (iter != session_map_.end()) {
        session = iter->second;
    } else {
        SessionPtr sess = NewSession();
        session_map_.insert({session_id, sess});
        session = sess;
    }
    return;
}

ge::graphStatus SessionMgr::CreateSession(const uint64_t session_id) {
    GELOGD("Create session_id: [%llu]", session_id);
    const std::lock_guard<std::mutex> lock(session_mutex_);
    const auto iter = session_map_.find(session_id);
    if (iter != session_map_.end()) {
        GELOGE(ge::PARAM_INVALID, "session_id [%llu] has already exist.", session_id);
        REPORT_INNER_ERR_MSG("E39999", "session_id has already exist.");
        return ge::PARAM_INVALID;
    }
    session_map_.insert({session_id, NewSession()});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SessionMgr::GetSession(const uint64_t session_id, SessionPtr& session) {
    GELOGD("Get session_id: [%llu]", session_id);
    const std::lock_guard<std::mutex> lock(session_mutex_);
    const auto iter = session_map_.find(session_id);
    if (iter == session_map_.end()) {
        GELOGE(ge::PARAM_INVALID, "session_id [%llu] does not exist.", session_id);
        REPORT_INNER_ERR_MSG("E39999", "session_id does not exist.");
        return ge::PARAM_INVALID;
    }
    session = iter->second;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SessionMgr::DestroySession(const uint64_t session_id) {
    GELOGD("Destroy session_id: [%llu]", session_id);
    const std::lock_guard<std::mutex> lock(session_mutex_);
    const auto iter = session_map_.find(session_id);
    if (iter == session_map_.end()) {
        GELOGE(ge::PARAM_INVALID, "session_id [%llu] does not exist.", session_id);
        REPORT_INNER_ERR_MSG("E39999", "session_id does not exist.");
        return ge::PARAM_INVALID;
    }
    session_map_.erase(iter);
    return ge::GRAPH_SUCCESS;
}

void SessionMgr::GetRm(const uint64_t session_id, const uint64_t container_id, ResourceMgrPtr &out_rm) {
    GELOGD("session_id: [%llu], container_id: [%llu]", session_id, container_id);
    SessionPtr session;
    SessionMgr::GetInstance()->GetOrCreateSession(session_id, session);
    ResourceMgrPtr rm;
    session->GetOrCreateRm(container_id, rm);
    out_rm = rm;
    return;
}

Session::~Session() {
    const std::lock_guard<std::mutex> lock(mutex_);
    rm_map_.clear();
}

void Session::GetOrCreateRm(const uint64_t container_id, ResourceMgrPtr &rm) {
    GELOGD("GetOrCreateRm container_id: [%llu]", container_id);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = rm_map_.find(container_id);
    if (iter != rm_map_.end()) {
        rm = iter->second;
    } else {
        ResourceMgrPtr tmp_rm = NewRm();
        rm_map_.insert({container_id, tmp_rm});
        rm = tmp_rm;
    }
    return;
}

ge::graphStatus Session::CreateRm(const uint64_t container_id) {
    GELOGD("Create container_id: [%llu]", container_id);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = rm_map_.find(container_id);
    if (iter != rm_map_.end()) {
        GELOGE(ge::PARAM_INVALID, "container_id [%llu] has already exist.", container_id);
        REPORT_INNER_ERR_MSG("E39999", "container_id has already exist.");
        return ge::PARAM_INVALID;
    }
    rm_map_.insert({container_id, NewRm()});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Session::ClearRm(const uint64_t container_id) {
    GELOGD("Clear container_id: [%llu]", container_id);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = rm_map_.find(container_id);
    if (iter == rm_map_.end()) {
        GELOGE(ge::PARAM_INVALID, "container_id [%llu] does not exist.", container_id);
        REPORT_INNER_ERR_MSG("E39999", "container_id does not exist.");
        return ge::PARAM_INVALID;
    } else {
        auto rm = iter->second;
        rm->ClearAllResource();
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus Session::ClearAllRm() {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto &id : container_id_) {
        auto ret = ClearRm(id);
        if (ret != 0) {
            GELOGE(ge::PARAM_INVALID, "clear container_id [%llu] failed.", id);
            REPORT_INNER_ERR_MSG("E39999", "clear container_id failed.");
            return ge::PARAM_INVALID;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Session::GetRm(const uint64_t container_id, ResourceMgrPtr &rm) {
    GELOGD("Get container_id: [%llu]", container_id);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = rm_map_.find(container_id);
    if (iter == rm_map_.end()) {
        GELOGE(ge::PARAM_INVALID, "container_id [%llu] does not exist.", container_id);
        REPORT_INNER_ERR_MSG("E39999", "container_id does not exist.");
        return ge::PARAM_INVALID;
    }
    rm = iter->second;
    return ge::GRAPH_SUCCESS;
}
}