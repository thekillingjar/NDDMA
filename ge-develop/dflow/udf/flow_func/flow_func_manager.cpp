/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_manager.h"
#include "common/udf_log.h"
#include "mmpa/mmpa_api.h"
#include "flow_func/single_func_wrapper.h"
#include "flow_func/multi_func_wrapper.h"
#include "flow_func/flow_func_config_manager.h"

namespace FlowFunc {

int32_t MetaFlowFunc::ResetFlowFuncState() {
    UDF_LOG_WARN("method is not implemented.");
    return FLOW_FUNC_ERR_NOT_SUPPORT;
}

int32_t MetaMultiFunc::ResetFlowFuncState(const std::shared_ptr<MetaParams> &params) {
    (void)params;
    UDF_LOG_WARN("method is not implemented.");
    return FLOW_FUNC_ERR_NOT_SUPPORT;
}

bool RegisterFlowFunc(const char *flow_func_name, const FLOW_FUNC_CREATOR_FUNC &func) noexcept {
    if (flow_func_name == nullptr) {
        UDF_LOG_ERROR("flow_func_name is null.");
        return false;
    }
    if (func == nullptr) {
        UDF_LOG_ERROR("func is null, flow_func_name=%s.", flow_func_name);
        return false;
    }
    FlowFuncManager::Instance().Register(flow_func_name, func);
    return true;
}

bool RegisterMultiFunc(const char *flow_func_name, const MULTI_FUNC_CREATOR_FUNC &func_creator) noexcept {
    if (flow_func_name == nullptr) {
        UDF_LOG_ERROR("flow_func_name is null.");
        return false;
    }
    if (func_creator == nullptr) {
        UDF_LOG_ERROR("func_creator is null, flow_func_name=%s.", flow_func_name);
        return false;
    }
    FlowFuncManager::Instance().Register(flow_func_name, func_creator);
    return true;
}

bool RegisterMultiFunc(const char *flow_func_name, const MULTI_FUNC_WITH_Q_CREATOR_FUNC &func_with_q_creator) noexcept {
    if (flow_func_name == nullptr) {
        UDF_LOG_ERROR("flow_func_name is null.");
        return false;
    }
    if (func_with_q_creator == nullptr) {
        UDF_LOG_ERROR("func_creator is null, flow_func_name=%s.", flow_func_name);
        return false;
    }
    FlowFuncManager::Instance().Register(flow_func_name, func_with_q_creator);
    return true;
}

FlowFuncManager::~FlowFuncManager() {
    Finalize();
}

FlowFuncManager &FlowFuncManager::Instance() {
    static FlowFuncManager inst;
    return inst;
}

std::shared_ptr<FuncWrapper> FlowFuncManager::TryGetFuncWrapper(const std::string &flow_func_name,
    const std::string &instance_name) {
    std::unique_lock<std::mutex> lock(guard_mutex_);
    std::map<std::string, FLOW_FUNC_CREATOR_FUNC>::const_iterator creator_iter = creator_map_.find(flow_func_name);
    if (creator_iter == creator_map_.cend()) {
        UDF_LOG_INFO("FlowFunc %s is not registered in single flow func.", flow_func_name.c_str());
        return nullptr;
    }
    std::shared_ptr<MetaFlowFunc> meta_flow_func;
    try {
        meta_flow_func = (creator_iter->second)();
    } catch (std::exception &e) {
        UDF_LOG_ERROR("Get FlowFunc %s exception, error=%s.", flow_func_name.c_str(), e.what());
        return nullptr;
    }
    lock.unlock();
    try {
        std::shared_ptr<FuncWrapper> func_wrapper = std::make_shared<SingleFuncWrapper>(meta_flow_func);
        std::unique_lock<std::mutex> wrapper_lock(func_wrapper_mutex_);
        func_wrapper_map_[instance_name][flow_func_name] = func_wrapper;
        return func_wrapper;
    } catch (std::exception &e) {
        UDF_LOG_ERROR("make SingleFuncWrapper %s exception, error=%s.", flow_func_name.c_str(), e.what());
    }
    return nullptr;
}

std::shared_ptr<FuncWrapper> FlowFuncManager::TryGetFuncWrapperFromMap(
    const std::string &flow_func_name, const std::string &instance_name) {
    std::unique_lock<std::mutex> lock(func_wrapper_mutex_);
    const auto instance_iter = func_wrapper_map_.find(instance_name);
    if (instance_iter == func_wrapper_map_.cend()) {
        UDF_LOG_DEBUG(
            "instance_name:%s is not in funcWrapperMap, flow_func_name=%s.", instance_name.c_str(), flow_func_name.c_str());
        return nullptr;
    }
    const auto func_wrapper_iter = instance_iter->second.find(flow_func_name);
    if (func_wrapper_iter == instance_iter->second.cend()) {
        UDF_LOG_DEBUG(
            "flow_func_name:%s is not in funcWrapperMap, instance_name=%s.", flow_func_name.c_str(), instance_name.c_str());
        return nullptr;
    }
    return func_wrapper_iter->second;
}

int32_t FlowFuncManager::TryCreateMultiFunc(const std::string &flow_func_name, const std::string &instance_name) {
    std::shared_ptr<MetaMultiFunc> multi_func;
    std::map<AscendString, PROC_FUNC_WITH_CONTEXT> proc_func_list;
    int32_t ret = FLOW_FUNC_SUCCESS;
    {
        std::unique_lock<std::mutex> lock(guard_mutex_);
        std::map<std::string, MULTI_FUNC_CREATOR_FUNC>::const_iterator creator_iter =
            multi_func_creator_map_.find(flow_func_name);
        if (creator_iter == multi_func_creator_map_.cend()) {
            UDF_LOG_INFO("FlowFunc %s is not registered in multi flow func.", flow_func_name.c_str());
            return FLOW_FUNC_SUCCESS;
        }
        {
            std::unique_lock<std::mutex> multi_func_inst_map_lock(multi_func_inst_mutex_);
            auto iter = multi_func_inst_map_.find(instance_name);
            if (iter != multi_func_inst_map_.end()) {
                multi_func = iter->second;
            }
        }
        try {
            ret = (creator_iter->second)(multi_func, proc_func_list);
        } catch (std::exception &e) {
            UDF_LOG_ERROR("create FlowFunc %s exception, error=%s.", flow_func_name.c_str(), e.what());
            return FLOW_FUNC_FAILED;
        }
    }
    if ((ret != FLOW_FUNC_SUCCESS) || (multi_func == nullptr)) {
        UDF_LOG_ERROR("create FlowFunc %s failed, ret=%d.", flow_func_name.c_str(), ret);
        return FLOW_FUNC_FAILED;
    }

    if (proc_func_list.count(flow_func_name.c_str()) == 0U) {
        UDF_LOG_ERROR("Create FlowFunc %s, but proc func does not contain it, proc_func_list.size=%zu.",
            flow_func_name.c_str(),
            proc_func_list.size());
        return FLOW_FUNC_FAILED;
    }
    {
        std::unique_lock<std::mutex> multi_func_inst_map_lock(multi_func_inst_mutex_);
        multi_func_inst_map_[instance_name] = multi_func;
    }
    
    std::unique_lock<std::mutex> func_wrapper_map_lock(func_wrapper_mutex_);
    for (const auto &proc_func : proc_func_list) {
        PROC_FUNC_WITH_CONTEXT func_with_context = proc_func.second;
        try {
            std::shared_ptr<FuncWrapper> func_wrapper = std::make_shared<MultiFuncWrapper>(multi_func, func_with_context);
            func_wrapper_map_[instance_name][proc_func.first.GetString()] = func_wrapper;
            UDF_LOG_DEBUG(
                "create FlowFunc %s for instance %s success.", proc_func.first.GetString(), instance_name.c_str());
        } catch (std::exception &e) {
            UDF_LOG_ERROR("make MultiFuncWrapper %s exception, error=%s.", flow_func_name.c_str(), e.what());
            return FLOW_FUNC_FAILED;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncManager::TryCreateMultiFuncWithQ(const std::string &flow_func_name, const std::string &instance_name) {
    std::shared_ptr<MetaMultiFunc> multi_func;
    std::map<AscendString, PROC_FUNC_WITH_CONTEXT_Q> proc_func_with_q_list;
    int32_t ret = FLOW_FUNC_SUCCESS;
    {
        std::unique_lock<std::mutex> lock(guard_mutex_);
        std::map<std::string, MULTI_FUNC_WITH_Q_CREATOR_FUNC>::const_iterator creator_iter =
            multi_func_with_q_creator_map_.find(flow_func_name);
        if (creator_iter == multi_func_with_q_creator_map_.cend()) {
            UDF_LOG_INFO("FlowFunc %s with input queues is not registered in multi flow func.", flow_func_name.c_str());
            return FLOW_FUNC_SUCCESS;
        }
        {
            std::unique_lock<std::mutex> multi_func_inst_map_lock(multi_func_inst_mutex_);
            auto iter = multi_func_inst_map_.find(instance_name);
            if (iter != multi_func_inst_map_.end()) {
                multi_func = iter->second;
            }
        }
        try {
            ret = (creator_iter->second)(multi_func, proc_func_with_q_list);
        } catch (std::exception &e) {
            UDF_LOG_ERROR("create FlowFunc %s with input queues exception, error=%s.", flow_func_name.c_str(), e.what());
            return FLOW_FUNC_FAILED;
        }
    }
    if ((ret != FLOW_FUNC_SUCCESS) || (multi_func == nullptr)) {
        UDF_LOG_ERROR("create FlowFunc %s with input queues failed, ret=%d.", flow_func_name.c_str(), ret);
        return FLOW_FUNC_FAILED;
    }

    if (proc_func_with_q_list.count(flow_func_name.c_str()) == 0U) {
        UDF_LOG_ERROR(
            "Create FlowFunc %s with input queues, but proc func does not contain it, proc_func_with_q_list.size=%zu.",
            flow_func_name.c_str(),
            proc_func_with_q_list.size());
        return FLOW_FUNC_FAILED;
    }
    {
        std::unique_lock<std::mutex> multi_func_inst_map_lock(multi_func_inst_mutex_);
        multi_func_inst_map_[instance_name] = multi_func;
    }

    std::unique_lock<std::mutex> func_wrapper_map_lock(func_wrapper_mutex_);
    for (const auto &proc_func : proc_func_with_q_list) {
        PROC_FUNC_WITH_CONTEXT_Q func_with_context = proc_func.second;
        try {
            std::shared_ptr<FuncWrapper> func_wrapper =
                std::make_shared<MultiFuncWrapper>(multi_func, nullptr, func_with_context);
            func_wrapper_map_[instance_name][proc_func.first.GetString()] = func_wrapper;
            UDF_LOG_DEBUG(
                "create FlowFunc %s for instance %s success.", proc_func.first.GetString(), instance_name.c_str());
        } catch (std::exception &e) {
            UDF_LOG_ERROR("make MultiFuncWrapper %s exception, error=%s.", flow_func_name.c_str(), e.what());
            return FLOW_FUNC_FAILED;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

std::shared_ptr<FuncWrapper> FlowFuncManager::TryGetMultiFuncWrapper(
    const std::string &flow_func_name, const std::string &instance_name) {
    auto func_wrapper = TryGetFuncWrapperFromMap(flow_func_name, instance_name);
    if (func_wrapper != nullptr) {
        return func_wrapper;
    }
    auto ret = TryCreateMultiFunc(flow_func_name, instance_name);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("create multi FlowFunc %s failed, instance_name=%s, ret=%d.",
            flow_func_name.c_str(),
            instance_name.c_str(),
            ret);
        return nullptr;
    }
    ret = TryCreateMultiFuncWithQ(flow_func_name, instance_name);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("create multi FlowFunc %s with input queues failed, instance_name=%s, ret=%d.",
            flow_func_name.c_str(),
            instance_name.c_str(),
            ret);
        return nullptr;
    }
    // after create, retry to get.
    return TryGetFuncWrapperFromMap(flow_func_name, instance_name);
}

std::shared_ptr<FuncWrapper> FlowFuncManager::GetFlowFuncWrapper(
    const std::string &flow_func_name, const std::string &instance_name) {
    auto func_wrapper = TryGetFuncWrapper(flow_func_name, instance_name);
    if (func_wrapper != nullptr) {
        UDF_LOG_INFO("get FlowFunc %s success, instance_name=%s.", flow_func_name.c_str(), instance_name.c_str());
        return func_wrapper;
    }
    func_wrapper = TryGetMultiFuncWrapper(flow_func_name, instance_name);
    if (func_wrapper != nullptr) {
        UDF_LOG_INFO(
            "get multi FlowFunc %s success, instance_name=%s.", flow_func_name.c_str(), instance_name.c_str());
        return func_wrapper;
    }
    UDF_LOG_ERROR(
        "get flow func wrapper failed, flow_func_name=%s, instance_name=%s.", flow_func_name.c_str(), instance_name.c_str());
    return nullptr;
}

void FlowFuncManager::Register(const std::string &flow_func_name, const FLOW_FUNC_CREATOR_FUNC &func) {
    std::unique_lock<std::mutex> lock(guard_mutex_);
    if (creator_map_.count(flow_func_name) != 0U) {
        UDF_LOG_ERROR("%s FlowFunc creator is already exist", flow_func_name.c_str());
        return;
    }
    creator_map_[flow_func_name] = func;
    UDF_RUN_LOG_INFO("%s register successful", flow_func_name.c_str());
}

void FlowFuncManager::Register(const std::string &flow_func_name, const MULTI_FUNC_CREATOR_FUNC &multi_func_creator) {
    std::unique_lock<std::mutex> lock(guard_mutex_);
    if (multi_func_creator_map_.count(flow_func_name) != 0U) {
        UDF_LOG_WARN("%s FlowFunc creator is already exist", flow_func_name.c_str());
        return;
    }
    multi_func_creator_map_[flow_func_name] = multi_func_creator;
    UDF_RUN_LOG_INFO("multi func %s register successful", flow_func_name.c_str());
}

void FlowFuncManager::Register(const std::string &flow_func_name, const MULTI_FUNC_WITH_Q_CREATOR_FUNC &multi_func_with_q_creator) {
    std::unique_lock<std::mutex> lock(guard_mutex_);
    if (multi_func_with_q_creator_map_.count(flow_func_name) != 0U) {
        UDF_LOG_WARN("%s FlowFunc with input queues creator is already exist", flow_func_name.c_str());
        return;
    }
    multi_func_with_q_creator_map_[flow_func_name] = multi_func_with_q_creator;
    UDF_RUN_LOG_INFO("multi func with input queues %s register successful", flow_func_name.c_str());
}

int32_t FlowFuncManager::LoadLib(const std::string &lib_path) {
    return ExecuteByAsyncThread([this, &lib_path]() {
        DoLoadLib(lib_path);
        return FLOW_FUNC_SUCCESS;
    });
}

int32_t FlowFuncManager::DoLoadLib(const std::string &lib_path) {
    constexpr auto mode = static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
                                               static_cast<uint32_t>(MMPA_RTLD_GLOBAL));
    void *handle = mmDlopen(lib_path.c_str(), mode);
    if (handle == nullptr) {
        const char *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        UDF_LOG_ERROR("load lib failed, lib_path=%s, error msg=%s.", lib_path.c_str(), error);
        return FLOW_FUNC_FAILED;
    }
    {
        std::unique_lock<std::mutex> lock(guard_mutex_);
        handle_map_.emplace(lib_path, handle);
    }
    UDF_LOG_INFO("load lib[%s] end.", lib_path.c_str());
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncManager::Init() {
    auto init_func = FlowFuncConfigManager::GetConfig()->GetLimitThreadInitFunc();
    // single thread load and unload so
    async_so_handler_.reset(new(std::nothrow) AsyncExecutor("udf_so_load_", 1, init_func));
    if (async_so_handler_ == nullptr) {
        UDF_LOG_ERROR("flow func manager init failed, as new async executor failed.");
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncManager::Reset() {
    (void)ExecuteByAsyncThread([this]() {
        {
            std::unique_lock<std::mutex> lock(func_wrapper_mutex_);
            func_wrapper_map_.clear();
        }
        {
            std::unique_lock<std::mutex> lock(multi_func_inst_mutex_);
            multi_func_inst_map_.clear();
        }
        return FLOW_FUNC_SUCCESS;
    });
}

int32_t FlowFuncManager::ResetFuncState() {
    return ExecuteByAsyncThread([this]() {
        std::unique_lock<std::mutex> lock(func_wrapper_mutex_);
        for (const auto &instance_func : func_wrapper_map_) {
            if (!instance_func.second.empty()) {
                const auto &func_wrapper = instance_func.second.cbegin();
                // just need reset once for an instance
                int32_t reset_ret = func_wrapper->second->ResetFlowFuncState();
                if (reset_ret != FLOW_FUNC_SUCCESS) {
                    if (reset_ret == FLOW_FUNC_ERR_NOT_SUPPORT) {
                        UDF_LOG_WARN("flow func not support reset state, instance name=%s, func name=%s",
                            instance_func.first.c_str(), func_wrapper->first.c_str());
                    } else {
                        UDF_LOG_ERROR("reset flow func state failed, instance name=%s, func name=%s, ret=%d",
                            instance_func.first.c_str(), func_wrapper->first.c_str(), reset_ret);
                    }
                    return reset_ret;
                }
            }
        }
        return FLOW_FUNC_SUCCESS;
    });
}

void FlowFuncManager::Finalize() noexcept {
    (void)ExecuteByAsyncThread([this]() {
        DoUnloadLib();
        return FLOW_FUNC_SUCCESS;
    });
}

int32_t FlowFuncManager::ExecuteByAsyncThread(const std::function<int32_t()> &exec_func) {
    if (async_so_handler_ == nullptr) {
        return exec_func();
    } else {
        auto fut = async_so_handler_->Commit([exec_func]() { return exec_func(); });
        try {
            return fut.get();
        } catch (const std::exception &ex) {
            UDF_LOG_ERROR("execute by async thread exception, error=%s.", ex.what());
            return FLOW_FUNC_FAILED;
        }
    }
}

void FlowFuncManager::DoUnloadLib() noexcept {
    {
        std::unique_lock<std::mutex> lock(func_wrapper_mutex_);
        func_wrapper_map_.clear();
    }
    {
        std::unique_lock<std::mutex> lock(multi_func_inst_mutex_);
        multi_func_inst_map_.clear();
    }
    std::unique_lock<std::mutex> lock(guard_mutex_);
    creator_map_.clear();
    multi_func_creator_map_.clear();
    multi_func_with_q_creator_map_.clear();
    for (const auto &handleIter : handle_map_) {
        UDF_LOG_INFO("unload lib[%s] start.", handleIter.first.c_str());
        int32_t closeRet = mmDlclose(handleIter.second);
        UDF_LOG_INFO("unload lib[%s] end, closeRet=%d.", handleIter.first.c_str(), closeRet);
    }
    handle_map_.clear();
}
}
