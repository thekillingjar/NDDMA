/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "multi_func_wrapper.h"
#include <mutex>
#include "common/inner_error_codes.h"
#include "common/udf_log.h"

namespace FlowFunc {
namespace {
std::mutex g_multi_func_init_mutex;
// key:flow func ptr, value:init ref times
std::map<const void *, int32_t> g_init_multi_func_list;
}  // namespace

MultiFuncWrapper::~MultiFuncWrapper() noexcept {
    std::lock_guard<std::mutex> lock_guard(g_multi_func_init_mutex);
    auto find_ret = g_init_multi_func_list.find(multi_func_.get());
    if (find_ret != g_init_multi_func_list.end()) {
        if (find_ret->second > 0) {
            --(find_ret->second);
        }
        if (find_ret->second == 0) {
            (void)g_init_multi_func_list.erase(find_ret);
        }
    }
}

int32_t MultiFuncWrapper::Init(
    const std::shared_ptr<MetaParams> &meta_param, const std::shared_ptr<MetaRunContext> &run_context) {
    {
        std::lock_guard<std::mutex> lock_guard(g_multi_func_init_mutex);
        // if not init, init it.
        if (g_init_multi_func_list[multi_func_.get()] == 0) {
            auto ret = multi_func_->Init(meta_param);
            if (ret == FLOW_FUNC_ERR_INIT_AGAIN) {
                // log outside
                return ret;
            }
            if (ret != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("multi_func init failed, ret=%d.", ret);
                return ret;
            }
        }
        ++g_init_multi_func_list[multi_func_.get()];
    }
    meta_param_ = meta_param;
    run_context_ = run_context;
    return FLOW_FUNC_SUCCESS;
}

int32_t MultiFuncWrapper::Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    if (proc_func_ == nullptr) {
        UDF_LOG_ERROR("proc_func is null.");
        return FLOW_FUNC_FAILED;
    }
    return proc_func_(run_context_, input_msgs);
}

int32_t MultiFuncWrapper::ResetFlowFuncState() {
    return multi_func_->ResetFlowFuncState(meta_param_);
}

int32_t MultiFuncWrapper::Proc(const std::vector<std::shared_ptr<FlowMsgQueue>> &input_queues) {
    if (proc_func_with_q_ == nullptr) {
        UDF_LOG_ERROR("proc_func_with_q is null.");
        return FLOW_FUNC_FAILED;
    }
    return proc_func_with_q_(run_context_, input_queues);
}
}  // namespace FlowFunc
