/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_engine_stub.h"
#include <mutex>
#include "common/aicpu_ops_kernel_info_store/aicpu_ops_kernel_info_store.h"
#include "common/aicpu_graph_optimizer/aicpu_graph_optimizer.h"
#include "common/aicpu_ops_kernel_builder/aicpu_ops_kernel_builder.h"
#include "util/log.h"

using namespace std;
namespace {
    const string OPS_KERNEL_INFO = "aicpu_kernel";
    const string GRAPH_OPTIMIZER = "aicpu_optimizer";
    const string KERNEL_BUILDER = "aicpu_builder";
}

namespace aicpu {
BaseEnginePtr BaseEngineStub::instance_ = nullptr;

BaseEnginePtr BaseEngineStub::Instance()
{
    static once_flag flag;
    call_once(flag, [&]() {
        instance_.reset(new (std::nothrow) BaseEngineStub);
    });
    return instance_;
}

ge::Status BaseEngineStub::Initialize(const map<string, string> &options)
{
    if (ops_kernel_info_store_ == nullptr) {
        ops_kernel_info_store_ = make_shared<AicpuOpsKernelInfoStore>("BaseEngineStub");
        if (ops_kernel_info_store_ == nullptr) {
            AICPUE_LOGE("Make AicpuOpsKernelInfoStore failed.");
            return ge::FAILED;
        }
    }
    if (graph_optimizer_ == nullptr) {
        graph_optimizer_ = make_shared<AicpuGraphOptimizer>("BaseEngineStub");
        if (graph_optimizer_ == nullptr) {
            AICPUE_LOGE("Make AicpuGraphOptimizer failed.");
            return ge::FAILED;
        }
    }
    if (ops_kernel_builder_ == nullptr) {
        ops_kernel_builder_ = make_shared<AicpuOpsKernelBuilder>("BaseEngineStub");
        if (ops_kernel_builder_ == nullptr) {
            AICPUE_LOGE("Make AicpuOpsKernelBuilder failed.");
            return ge::FAILED;
        }
    }
    BaseEnginePtr (* instancePtr)() = &BaseEngineStub::Instance;
    return LoadConfigFile(reinterpret_cast<void*>(instancePtr));
}

void BaseEngineStub::GetOpsKernelInfoStores(map<string, OpsKernelInfoStorePtr> &opsKernelInfoStores) const
{
    if (ops_kernel_info_store_ != nullptr) {
        opsKernelInfoStores[OPS_KERNEL_INFO] = ops_kernel_info_store_;
    }
}

void BaseEngineStub::GetGraphOptimizerObjs(map<string, GraphOptimizerPtr> &graphOptimizers) const
{
    if (graph_optimizer_ != nullptr) {
        graphOptimizers[GRAPH_OPTIMIZER] = graph_optimizer_;
    }
}

void BaseEngineStub::GetOpsKernelBuilderObjs(map<string, OpsKernelBuilderPtr> &opsKernelBuilders) const
{
    if (ops_kernel_builder_ != nullptr) {
        opsKernelBuilders[KERNEL_BUILDER] = ops_kernel_builder_;
    }
}

ge::Status BaseEngineStub::Finalize()
{
    ops_kernel_info_store_ = nullptr;
    graph_optimizer_ = nullptr;
    ops_kernel_builder_ = nullptr;
    return ge::SUCCESS;
}

}  // namespace aicpu

ge::Status Initialize(const map<string, string> &options)
{
    return aicpu::BaseEngineStub::Instance()->Initialize(options);
}

void GetOpsKernelInfoStores(map<string, OpsKernelInfoStorePtr> &opsKernelInfoStores)
{
    aicpu::BaseEngineStub::Instance()->GetOpsKernelInfoStores(opsKernelInfoStores);
}

void GetGraphOptimizerObjs(map<string, GraphOptimizerPtr> &graphOptimizers)
{
    aicpu::BaseEngineStub::Instance()->GetGraphOptimizerObjs(graphOptimizers);
}

void GetOpsKernelBuilderObjs(map<string, OpsKernelBuilderPtr> &opsKernelBuilders)
{
    aicpu::BaseEngineStub::Instance()->GetOpsKernelBuilderObjs(opsKernelBuilders);
}

ge::Status Finalize()
{
    return aicpu::BaseEngineStub::Instance()->Finalize();
}
