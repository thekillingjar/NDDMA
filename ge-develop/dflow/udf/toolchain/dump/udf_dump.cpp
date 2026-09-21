/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_dump.h"
#include <mutex>
#include "common/udf_log.h"
#include "udf_dump_manager.h"
#ifdef RUN_ON_DEVICE
#include "udf_dump_task_device.h"
#else
#include "udf_dump_task_host.h"
#endif

namespace FlowFunc {
namespace {
std::once_flag g_dumpInit;
}
int32_t UdfDump::DumpInput(const std::string &flow_func_info, const std::vector<MbufTensor *> &tensors,
                           const uint32_t step_id) {
    if (!UdfDumpManager::Instance().IsEnableDump() || (UdfDumpManager::Instance().GetDumpMode() == "output") ||
        !UdfDumpManager::Instance().IsInDumpStep(step_id)) {
        UDF_LOG_INFO("No need dump input.");
        return FLOW_FUNC_SUCCESS;
    }

    if (tensors.empty()) {
        UDF_LOG_WARN("No input tensor exists.");
        return FLOW_FUNC_SUCCESS;
    }

#ifdef RUN_ON_DEVICE
    std::call_once(g_dumpInit, []() {
        auto initRet = UdfDumpManager::Instance().InitAicpuDump();
        UDF_LOG_INFO("InitAicpuDump ret=%d.", initRet);
    });
    UdfDumpTaskDevice dumpTask(flow_func_info, UdfDumpManager::Instance().GetHostPid(),
                               UdfDumpManager::Instance().GetDeviceId(), UdfDumpManager::Instance().GetAicpuPid(),
                               UdfDumpManager::Instance().GetLogicDeviceId());
#else
    UdfDumpTaskHost dumpTask(flow_func_info, UdfDumpManager::Instance().GetHostPid(),
                              UdfDumpManager::Instance().GetDeviceId());
#endif

    auto ret = dumpTask.PreProcessInput(tensors);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("PreProcessInput failed, ret=%d.", ret);
        return ret;
    }

    ret = dumpTask.DumpOpInfo(UdfDumpManager::Instance().GetDumpPath(), step_id);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("DumpOpInfo failed, ret=%d.", ret);
        return ret;
    }

    return FLOW_FUNC_SUCCESS;
}

int32_t UdfDump::DumpOutput(const std::string &flow_func_info, uint32_t index, const MbufTensor *tensor,
                            const uint32_t step_id) {
    // dump output
    if (!UdfDumpManager::Instance().IsEnableDump() || (UdfDumpManager::Instance().GetDumpMode() == "input") ||
        !UdfDumpManager::Instance().IsInDumpStep(step_id)) {
        UDF_LOG_INFO("No need dump output.");
        return FLOW_FUNC_SUCCESS;
    }

#ifdef RUN_ON_DEVICE
    std::call_once(g_dumpInit, []() {
        auto initRet = UdfDumpManager::Instance().InitAicpuDump();
        UDF_LOG_INFO("InitAicpuDump ret=%d.", initRet);
    });
    UdfDumpTaskDevice dumpTask(flow_func_info, UdfDumpManager::Instance().GetHostPid(),
                               UdfDumpManager::Instance().GetDeviceId(), UdfDumpManager::Instance().GetAicpuPid(),
                               UdfDumpManager::Instance().GetLogicDeviceId());
#else
    UdfDumpTaskHost dumpTask(flow_func_info, UdfDumpManager::Instance().GetHostPid(),
                               UdfDumpManager::Instance().GetDeviceId());
#endif

    auto ret = dumpTask.PreProcessOutput(index, tensor);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("PreProcessOutput failed, ret=%d.", ret);
        return ret;
    }

    ret = dumpTask.DumpOpInfo(UdfDumpManager::Instance().GetDumpPath(), step_id);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("DumpOpInfo failed, ret=%d.", ret);
        return ret;
    }
    return FLOW_FUNC_SUCCESS;
}
}  // namespace FlowFunc
