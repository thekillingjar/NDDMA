/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_params.h"
#include "flow_func_manager.h"
#include "common/udf_log.h"

namespace FlowFunc {
namespace {
constexpr const char *kAttrNameBalanceScatter = "_balance_scatter";
constexpr const char *kAttrNameBalanceGather = "_balance_gather";
}

int32_t MetaParams::GetRunningInstanceId() const {
    UDF_LOG_ERROR("Not supported, please implement the GetRunningInstanceId method.");
    return -1;
}

int32_t MetaParams::GetRunningInstanceNum() const {
    UDF_LOG_ERROR("Not supported, please implement the GetRunningInstanceNum method.");
    return -1;
}

FlowFuncParams::FlowFuncParams(const std::string &name, size_t input_queue_num, size_t output_queue_num,
    int32_t running_device_id, int32_t device_id)
    : MetaParams(), name_(name), output_queued_locks_(output_queue_num), input_queue_num_(input_queue_num),
    output_queue_num_(output_queue_num), running_device_id_(running_device_id), device_id_(static_cast<uint32_t>(device_id)){
    }

FlowFuncParams::~FlowFuncParams() {
    output_queued_locks_.clear();
}

std::shared_ptr<const AttrValue> FlowFuncParams::GetAttr(const char *attr_name) const {
    if (attr_name == nullptr) {
        UDF_LOG_INFO("attr_name is null, instance name[%s].", name_.c_str());
        return nullptr;
    }
    UDF_LOG_DEBUG("get attr, instance name[%s], attr_name=%s.", name_.c_str(), attr_name);
    const auto attr_iter = attr_map_.find(attr_name);
    if (attr_iter == attr_map_.cend()) {
        UDF_LOG_INFO("no attr found, instance name[%s], attr_name=%s.", name_.c_str(), attr_name);
        return nullptr;
    }
    return attr_iter->second;
}

void FlowFuncParams::AddFlowModel(const std::string &model_key, std::unique_ptr<FlowModel> &&flow_model) {
    flow_models_[model_key] = std::move(flow_model);
}

std::mutex &FlowFuncParams::GetOutputQueueLocks(uint32_t output_idx) {
    if (output_idx < output_queued_locks_.size()) {
        return output_queued_locks_[output_idx];
    }
    static std::mutex default_mutex;
    return default_mutex;
}

int32_t FlowFuncParams::InitBalanceAttr() {
    const auto scatter_attr = GetAttr(kAttrNameBalanceScatter);
    if (scatter_attr != nullptr) {
        (void)scatter_attr->GetVal(is_balance_scatter_);
    }
    const auto gather_attr = GetAttr(kAttrNameBalanceGather);
    if (gather_attr != nullptr) {
        (void)gather_attr->GetVal(is_balance_gather_);
    }
    if (is_balance_scatter_ && is_balance_gather_) {
        UDF_LOG_ERROR("Init balance attr failed, as cannot set balance scatter and gather attr both, instance name[%s]",
            name_.c_str());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    UDF_LOG_INFO("init balance attr end, isBalanceScatter=%d, isBalanceGather=%d, instance name[%s]",
        is_balance_scatter_, is_balance_gather_, name_.c_str());
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncParams::Init() {
    if (is_inited_) {
        return FLOW_FUNC_SUCCESS;
    }

    int32_t ret = InitBalanceAttr();
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("init balance attr failed, instance name[%s], ret=%d.", name_.c_str(), ret);
        return ret;
    }

    ret = FlowFuncManager::Instance().LoadLib(lib_path_);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("load FlowFunc lib failed, instance name[%s], ret=%d.", name_.c_str(), ret);
        return ret;
    }

    for (auto &flow_model : flow_models_) {
        ret = flow_model.second->Init();
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("FlowModelImpl init failed, instance name[%s], model_key=%s.", name_.c_str(),
                flow_model.first.c_str());
            return ret;
        }
    }
    is_inited_ = true;
    UDF_RUN_LOG_INFO("FlowFunc params init success. instance name[%s].", name_.c_str());
    return FLOW_FUNC_SUCCESS;
}

FlowModel *FlowFuncParams::GetFlowModels(const std::string &model_key) const {
    auto flow_fun_iter = flow_models_.find(model_key);
    if (flow_fun_iter == flow_models_.cend()) {
        UDF_LOG_ERROR("no flow model found, instance name[%s], model_key=%s.", name_.c_str(), model_key.c_str());
        return nullptr;
    }
    return flow_fun_iter->second.get();
}

bool FlowFuncParams::HasInvokedModel(const std::string &scope, std::string &key) const {
    for (const std::string &invoked_scope : invoked_scopes_) {
        UDF_LOG_DEBUG("Exception scope[%s] and current invoked scope[%s].", scope.c_str(), invoked_scope.c_str());
        if ((invoked_scope.size() >= scope.size()) && (invoked_scope.compare(0, scope.size(), scope) == 0)) {
            const std::string model_key = invoked_scope.substr(scope.size());
            if (flow_models_.count(model_key) > 0UL) {
                key = model_key;
                return true;
            }
        }
    }
    UDF_LOG_INFO("no flow model found, scope=%s.", scope.c_str());
    return false;
}

bool FlowFuncParams::HandleInvokedException(const std::string &scope, uint64_t trans_id, bool is_add_exception) const {
    if (scope.empty()) {
        return false;
    }
    std::string model_key;
    if (HasInvokedModel(scope, model_key)) {
        auto flow_model = GetFlowModels(model_key);
        if (flow_model == nullptr) {
            return false;
        }
        if (is_add_exception) {
            flow_model->AddExceptionTransId(trans_id);
        } else {
            flow_model->DeleteExceptionTransId(trans_id);
        }
        return true;
    }
    return false;
}
}  // namespace FlowFunc