/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_META_MULTI_FUNC_H
#define FLOW_FUNC_META_MULTI_FUNC_H

#include <functional>
#include <map>
#include "flow_func_defines.h"
#include "meta_run_context.h"
#include "meta_params.h"
#include "flow_msg.h"
#include "flow_msg_queue.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaMultiFunc {
public:
    MetaMultiFunc() = default;

    virtual ~MetaMultiFunc() = default;

    /**
     * @brief multi func init.
     * @return 0:success, other: failed.
     */
    virtual int32_t Init(const std::shared_ptr<MetaParams> &params)
    {
        (void)params;
        return FLOW_FUNC_SUCCESS;
    }

    virtual int32_t ResetFlowFuncState(const std::shared_ptr<MetaParams> &params);
};

using PROC_FUNC_WITH_CONTEXT =
    std::function<int32_t(const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsg>> &)>;

using PROC_FUNC_WITH_CONTEXT_Q = 
    std::function<int32_t(const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsgQueue>> &)>;

using MULTI_FUNC_CREATOR_FUNC = std::function<int32_t(
    std::shared_ptr<MetaMultiFunc> &multiFunc, std::map<AscendString, PROC_FUNC_WITH_CONTEXT> &procFuncList)>;

using MULTI_FUNC_WITH_Q_CREATOR_FUNC = std::function<int32_t(
    std::shared_ptr<MetaMultiFunc> &multiFunc, std::map<AscendString, PROC_FUNC_WITH_CONTEXT_Q> &procFuncList)>;

/**
 * @brief register multi func creator.
 * @param flowFuncName cannot be null and must end with '\0'
 * @param func_creator multi func creator
 * @return if register success, true:success.
 */
FLOW_FUNC_VISIBILITY bool RegisterMultiFunc(
    const char *flowFuncName, const MULTI_FUNC_CREATOR_FUNC &funcCreator) noexcept;

FLOW_FUNC_VISIBILITY bool RegisterMultiFunc(
    const char *flowFuncName, const MULTI_FUNC_WITH_Q_CREATOR_FUNC &funcWithQCreator) noexcept;

template <typename T>
class FlowFuncRegistrar {
public:
    using CUSTOM_PROC_FUNC = std::function<int32_t(
        T *, const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsg>> &)>;

    using CUSTOM_PROC_FUNC_WITH_Q = std::function<int32_t(
        T *, const std::shared_ptr<MetaRunContext> &, const std::vector<std::shared_ptr<FlowMsgQueue>> &)>;

    FlowFuncRegistrar &RegProcFunc(const char *flowFuncName, const CUSTOM_PROC_FUNC &func)
    {
        using namespace std::placeholders;
        funcMap_[flowFuncName] = func;
        (void)RegisterMultiFunc(flowFuncName, std::bind(&FlowFuncRegistrar::CreateMultiFunc, this, _1, _2));
        return *this;
    }

    FlowFuncRegistrar &RegProcFunc(const char *flowFuncName, const CUSTOM_PROC_FUNC_WITH_Q &func)
    {
        using namespace std::placeholders;
        funcWithQMap_[flowFuncName] = func;
        (void)RegisterMultiFunc(flowFuncName, std::bind(&FlowFuncRegistrar::CreateMultiFuncWithQ, this, _1, _2));
        return *this;
    }

    int32_t CreateMultiFunc(std::shared_ptr<MetaMultiFunc> &multiFunc,
        std::map<AscendString, PROC_FUNC_WITH_CONTEXT> &procFuncMap) const
    {
        using namespace std::placeholders;
        T *flowFuncPtr = nullptr;
        if (GetFlowFuncInstance(multiFunc, flowFuncPtr) != FLOW_FUNC_SUCCESS) {
            return FLOW_FUNC_FAILED;
        }
        for (const auto &func : funcMap_) {
            procFuncMap[func.first] = std::bind(func.second, flowFuncPtr, _1, _2);
        }
        return FLOW_FUNC_SUCCESS;
    }

    int32_t CreateMultiFuncWithQ(std::shared_ptr<MetaMultiFunc> &multiFunc,
        std::map<AscendString, PROC_FUNC_WITH_CONTEXT_Q> &procFuncWithQMap) const
    {
        using namespace std::placeholders;
        T *flowFuncPtr = nullptr;
        if (GetFlowFuncInstance(multiFunc, flowFuncPtr) != FLOW_FUNC_SUCCESS) {
            return FLOW_FUNC_FAILED;
        }
        for (const auto &func : funcWithQMap_) {
            procFuncWithQMap[func.first] = std::bind(func.second, flowFuncPtr, _1, _2);
        }
        return FLOW_FUNC_SUCCESS;
    }

private:
    int32_t GetFlowFuncInstance(std::shared_ptr<MetaMultiFunc> &multiFunc, T *&flowFuncPtr) const
    {
        if (multiFunc == nullptr) {
            flowFuncPtr = new(std::nothrow) T();
            if (flowFuncPtr == nullptr) {
                return FLOW_FUNC_FAILED;
            }
            multiFunc.reset(flowFuncPtr);
        } else {
            flowFuncPtr = dynamic_cast<T *>(multiFunc.get());
            if (flowFuncPtr == nullptr) {
                return FLOW_FUNC_FAILED;
            }
        }
        return FLOW_FUNC_SUCCESS;
    }

    std::map<AscendString, CUSTOM_PROC_FUNC> funcMap_;
    std::map<AscendString, CUSTOM_PROC_FUNC_WITH_Q> funcWithQMap_;
};
/**
 * @brief define flow func REGISTRAR.
 * example:
 * FLOW_FUNC_REGISTRAR(UserFlowFunc).RegProcFunc("xxx_func", &UserFlowFunc::Proc1).
 *         RegProcFunc("xxx_func", &UserFlowFunc::Proc2);
 */
#define FLOW_FUNC_REGISTRAR(clazz)                                          \
    static FlowFunc::FlowFuncRegistrar<clazz> g_##clazz##FlowFuncRegistrar; \
    static auto &g_##clazz##Registrar = g_##clazz##FlowFuncRegistrar
}  // namespace FlowFunc
#endif  // FLOW_FUNC_META_MULTI_FUNC_H
