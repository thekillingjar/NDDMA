/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_ENGINE_STUB_H
#define BASE_ENGINE_STUB_H

#include "common/engine/base_engine.h"

namespace aicpu {
using BaseEnginePtr = std::shared_ptr<BaseEngine>;

/**
 * tf engine.
 * Used for the ops belong to tf engine.
 */
class BaseEngineStub : public BaseEngine{
public:
    /**
     * get BaseEngineStub instance.
     * @return  BaseEngineStub instance.
     */
    static BaseEnginePtr Instance();

    virtual ~BaseEngineStub() = default;

    /**
     * When Ge start, GE will invoke this interface
     * @return The status whether initialize successfully
     */
    ge::Status Initialize(const std::map<std::string, std::string> &options) override;

    /**
     * After the initialize, GE will invoke this interface
     * to get the Ops kernel Store.
     * @param opsKernels The tf ops kernel info
     */
    void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &opsKernelInfoStores) const override;

    /**
     * After the initialize, GE will invoke this interface
     * to get the Graph Optimizer.
     * @param graph_optimizers The tf Graph Optimizer objs
     */
    void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graphOptimizers) const override;

    /**
     * After the initialize, GE will invoke this interface
     * to get the Graph Optimizer.
     * @param graph_optimizers The tf Graph Optimizer objs
     */
    void GetOpsKernelBuilderObjs(std::map<std::string, OpsKernelBuilderPtr> &opsKernelBuilders) const override;

    /**
     * When the graph finished, GE will invoke this interface
     * @return The status whether initialize successfully
     */
    ge::Status Finalize() override;

    // Copy prohibited
    BaseEngineStub(const BaseEngineStub &tfEngine) = delete;

    // Move prohibited
    BaseEngineStub(const BaseEngineStub &&tfEngine) = delete;

    // Copy prohibited
    BaseEngineStub &operator=(const BaseEngineStub &tfEngine) = delete;

    // Move prohibited
    BaseEngineStub &operator=(BaseEngineStub &&tfEngine) = delete;

private:
    BaseEngineStub() : BaseEngine("baseEngineStub") {}

private:
    static BaseEnginePtr instance_;
};

}  // namespace aicpu

#endif
