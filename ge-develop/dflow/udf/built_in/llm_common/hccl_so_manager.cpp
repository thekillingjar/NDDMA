/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llm_common/hccl_so_manager.h"
#include <dlfcn.h>
#include <vector>
#include "common/udf_log.h"
#include "flow_func/flow_func_log.h"

namespace FlowFunc {
namespace {
constexpr const char *kHcclSoName = "libhccd.so";
}

HcclSoManager *HcclSoManager::GetInstance()
{
    static HcclSoManager manager;
    return &manager;
}

int32_t HcclSoManager::LoadSo()
{
    if (so_handle_ != nullptr) {
        UDF_LOG_INFO("Hccl so:%s has already been loaded.", kHcclSoName);
        return FLOW_FUNC_SUCCESS;
    }
    so_handle_ = dlopen(kHcclSoName, RTLD_LAZY);
    if (so_handle_ == nullptr) {
        UDF_LOG_ERROR("Fail to load hccl so:%s, error:%s.", kHcclSoName, dlerror());
        return FLOW_FUNC_FAILED;
    }
    const std::vector<const char *> k_func_names = {kHcclRawAcceptFuncName,
                                                    kHcclRawConnectFuncName,
                                                    kHcclRawListenFuncName,
                                                    kHcclRawIsendFuncName,
                                                    kHcclRawImprobeFuncName,
                                                    kHcclRawImRecvFuncName,
                                                    kHcclRawImRecvScatterFuncName,
                                                    kHcclRawTestSomeFuncName,
                                                    kHcclRawGetCountFuncName,
                                                    kHcclRawOpenFuncName,
                                                    kHcclRawCloseFuncName,
                                                    kHcclRawForceCloseFuncName,
                                                    kHcclRawBindFuncName,
                                                    kHcclRegisterGlobalMemoryFuncName,
                                                    kHcclUnregisterGlobalMemoryFuncName};
    for (auto iter = k_func_names.begin(); iter != k_func_names.end(); ++iter) {
        void *func = dlsym(so_handle_, *iter);
        if (func == nullptr) {
            UDF_LOG_ERROR("Fail to get function:%s from hccl so:%s, error:%s.", *iter, kHcclSoName, dlerror());
            func_map_.clear();
            return FLOW_FUNC_FAILED;
        }
        UDF_LOG_INFO("Success to get function:%s from hccl so:%s.", *iter, kHcclSoName);
        (void) func_map_.insert(std::make_pair(*iter, func));
    }
    UDF_LOG_INFO("Success to load hccl so:%s.", kHcclSoName);
    return FLOW_FUNC_SUCCESS;
};

int32_t HcclSoManager::UnloadSo()
{
    func_map_.clear();
    if (so_handle_ != nullptr) {
        (void) dlclose(so_handle_);
        so_handle_ = nullptr;
        UDF_LOG_INFO("Success to unload hccl so:%s.", kHcclSoName);
    }
    return FLOW_FUNC_SUCCESS;
}

void *HcclSoManager::GetFunc(const std::string &name) const
{
    const auto k_iter = func_map_.find(name);
    if (k_iter != func_map_.end()) {
        return k_iter->second;
    }
    return nullptr;
}

HcclSoManager::~HcclSoManager()
{
    (void) UnloadSo();
}
}  // namespace FlowFunc