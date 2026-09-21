/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_MANAGER_H
#define FLOW_FUNC_MANAGER_H

#include <map>
#include <mutex>
#include "flow_func/flow_func_defines.h"
#include "flow_func/meta_flow_func.h"
#include "flow_func/meta_multi_func.h"
#include "flow_func/func_wrapper.h"
#include "async_executor.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowFuncManager {
public:
    static FlowFuncManager &Instance();

    std::shared_ptr<FuncWrapper> GetFlowFuncWrapper(const std::string &flow_func_name, const std::string &instance_name);

    void Register(const std::string &flow_func_name, const FLOW_FUNC_CREATOR_FUNC &func);

    void Register(const std::string &flow_func_name, const MULTI_FUNC_CREATOR_FUNC &multi_func_creator);

    void Register(const std::string &flow_func_name, const MULTI_FUNC_WITH_Q_CREATOR_FUNC &multi_func_with_q_creator);

    int32_t LoadLib(const std::string &lib_path);

    void Reset();

    int32_t Init();

    int32_t ExecuteByAsyncThread(const std::function<int32_t (void)> &exec_func);

    int32_t ResetFuncState();

private:
    FlowFuncManager() = default;

    ~FlowFuncManager();

    int32_t DoLoadLib(const std::string &lib_path);

    void DoUnloadLib() noexcept;

    std::shared_ptr<FuncWrapper> TryGetFuncWrapper(const std::string &flow_func_name, const std::string &instance_name);

    std::shared_ptr<FuncWrapper> TryGetMultiFuncWrapper(
        const std::string &flow_func_name, const std::string &instance_name);

    std::shared_ptr<FuncWrapper> TryGetFuncWrapperFromMap(
        const std::string &flow_func_name, const std::string &instance_name);

    int32_t TryCreateMultiFunc(const std::string &flow_func_name, const std::string &instance_name);

    int32_t TryCreateMultiFuncWithQ(const std::string &flow_func_name, const std::string &instance_name);

    void Finalize() noexcept;

    std::map<std::string, FLOW_FUNC_CREATOR_FUNC> creator_map_;
    std::map<std::string, MULTI_FUNC_CREATOR_FUNC> multi_func_creator_map_;
    std::map<std::string, MULTI_FUNC_WITH_Q_CREATOR_FUNC> multi_func_with_q_creator_map_;
    std::map<std::string, void *> handle_map_;
    std::mutex guard_mutex_;

    std::mutex func_wrapper_mutex_;
    // key1: instance name, key2: func name, value: FuncWrapper
    std::map<std::string, std::map<std::string, std::shared_ptr<FuncWrapper>>> func_wrapper_map_;

    std::mutex multi_func_inst_mutex_;
    // key: instance name, value: MetaMultiFunc
    std::map<std::string, std::shared_ptr<MetaMultiFunc>> multi_func_inst_map_;
    // used for handle udf in one thread, as so load, unload, constructor, destructor, init
    std::unique_ptr<AsyncExecutor> async_so_handler_ = nullptr;
};
}
#endif // FLOW_FUNC_REGISTER_H
