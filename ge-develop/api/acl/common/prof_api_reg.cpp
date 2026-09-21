/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "prof_api_reg.h"
#include <mutex>
#include <unordered_set>
#include <map>
#include "mmpa/mmpa_api.h"
#include "common/log_inner.h"

namespace {
    static bool g_profRun = false;
    static std::mutex g_profMutex;
    static std::unordered_set<uint32_t> g_deviceList;
    constexpr uint64_t ACL_PROF_ACL_API = 0x0001U;
    constexpr uint32_t START_PROFILING = 1U;
    constexpr uint32_t STOP_PROFILING = 2U;

    static bool IsDumpToStdEnabled() {
        const char *profilingToStdOut = nullptr;
        MM_SYS_GET_ENV(MM_ENV_GE_PROFILING_TO_STD_OUT, profilingToStdOut);
        return profilingToStdOut != nullptr;
    }

    static const std::map<acl::AclProfType, std::string> PROF_TYPE_TO_NAMES = {
        {acl::AclProfType::AclopLoad,                                "aclopLoad"},
        {acl::AclProfType::AclopExecute,                             "aclopExecute"},
        {acl::AclProfType::AclopCreateHandle,                        "aclopCreateHandle"},
        {acl::AclProfType::AclopDestroyHandle,                       "aclopDestroyHandle"},
        {acl::AclProfType::AclopExecWithHandle,                      "aclopExecWithHandle"},
        {acl::AclProfType::AclopExecuteV2,                           "aclopExecuteV2"},
        {acl::AclProfType::AclopCreateKernel,                        "aclopCreateKernel"},
        {acl::AclProfType::AclopUpdateParams,                        "aclopUpdateParams"},
        {acl::AclProfType::AclopInferShape,                          "aclopInferShape"},
        {acl::AclProfType::AclopCast,                                "aclopCast"},
        {acl::AclProfType::AclopCreateHandleForCast,                 "aclopCreateHandleForCast"},
        {acl::AclProfType::AclopCreateAttr,                          "aclopCreateAttr"},
        {acl::AclProfType::AclopDestroyAttr,                         "aclopDestroyAttr"},
        {acl::AclProfType::AclopCompile,                             "aclopCompile"},
        {acl::AclProfType::AclopCompileAndExecute,                   "aclopCompileAndExecute"},
        {acl::AclProfType::AclopCompileAndExecuteV2,                 "aclopCompileAndExecuteV2"},
        {acl::AclProfType::AclGenGraphAndDumpForOp,                  "aclGenGraphAndDumpForOp"},
        {acl::AclProfType::OpCompile,                                "opCompile"},
        {acl::AclProfType::OpCompileAndDump,                         "opCompileAndDump"},
        {acl::AclProfType::AclmdlExecute,                            "aclmdlExecute"},
        {acl::AclProfType::AclmdlLoadFromMemWithQ,                   "aclmdlLoadFromMemWithQ"},
        {acl::AclProfType::AclmdlLoadFromMemWithMem,                 "aclmdlLoadFromMemWithMem"},
        {acl::AclProfType::AclmdlGetDesc,                            "aclmdlGetDesc"},
        {acl::AclProfType::AclmdlLoadFromFile,                       "aclmdlLoadFromFile"},
        {acl::AclProfType::AclmdlLoadFromFileWithMem,                "aclmdlLoadFromFileWithMem"},
        {acl::AclProfType::AclmdlLoadFromMem,                        "aclmdlLoadFromMem"},
        {acl::AclProfType::AclmdlBundleLoadFromFile,                 "aclmdlBundleLoadFromFile"},
        {acl::AclProfType::AclmdlBundleLoadFromMem,                  "aclmdlBundleLoadFromMem"},
        {acl::AclProfType::AclmdlBundleLoadModelWithMem,             "aclmdlBundleLoadModelWithMem"},
        {acl::AclProfType::AclmdlBundleLoadModelWithConfig,          "aclmdlBundleLoadModelWithConfig"},
        {acl::AclProfType::AclmdlBundleUnload,                       "aclmdlBundleUnload"},
        {acl::AclProfType::AclmdlBundleUnloadModel,                  "aclmdlBundleUnloadModel"},
        {acl::AclProfType::AclmdlSetInputAIPP,                       "aclmdlSetInputAIPP"},
        {acl::AclProfType::AclmdlSetAIPPByInputIndex,                "aclmdlSetAIPPByInputIndex"},
        {acl::AclProfType::AclmdlExecuteAsync,                       "aclmdlExecuteAsync"},
        {acl::AclProfType::AclmdlQuerySize,                          "aclmdlQuerySize"},
        {acl::AclProfType::AclmdlQuerySizeFromMem,                   "aclmdlQuerySizeFromMem"},
        {acl::AclProfType::AclmdlSetDynamicBatchSize,                "aclmdlSetDynamicBatchSize"},
        {acl::AclProfType::AclmdlSetDynamicHWSize,                   "aclmdlSetDynamicHWSize"},
        {acl::AclProfType::AclmdlSetInputDynamicDims,                "aclmdlSetInputDynamicDims"},
        {acl::AclProfType::AclmdlLoadWithConfig,                     "aclmdlLoadWithConfig"},
        {acl::AclProfType::AclmdlLoadFromFileWithQ,                  "aclmdlLoadFromFileWithQ"},
        {acl::AclProfType::AclmdlUnload,                             "aclmdlUnload"},
        {acl::AclProfType::AclCreateTensorDesc,                      "aclCreateTensorDesc"},
        {acl::AclProfType::AclDestroyTensorDesc,                     "aclDestroyTensorDesc"},
        {acl::AclProfType::AclTransTensorDescFormat,                 "aclTransTensorDescFormat"},
        {acl::AclProfType::AclblasGemmEx,                            "aclblasGemmEx"},
        {acl::AclProfType::AclblasCreateHandleForGemmEx,             "aclblasCreateHandleForGemmEx"},
        {acl::AclProfType::AclblasCreateHandleForHgemm,              "aclblasCreateHandleForHgemm"},
        {acl::AclProfType::AclblasHgemm,                             "aclblasHgemm"},
        {acl::AclProfType::AclblasS8gemm,                            "aclblasS8gemm"},
        {acl::AclProfType::AclblasCreateHandleForS8gemm,             "aclblasCreateHandleForS8gemm"},
        {acl::AclProfType::AclblasGemvEx,                            "aclblasGemvEx"},
        {acl::AclProfType::AclblasCreateHandleForGemvEx,             "aclblasCreateHandleForGemvEx"},
        {acl::AclProfType::AclblasHgemv,                             "aclblasHgemv"},
        {acl::AclProfType::AclblasCreateHandleForHgemv,              "aclblasCreateHandleForHgemv"},
        {acl::AclProfType::AclblasCreateHandleForS8gemv,             "aclblasCreateHandleForS8gemv"},
        {acl::AclProfType::AclblasS8gemv,                            "aclblasS8gemv"},
    };

    static aclError RegisterProfType() {
        for (auto &iter : PROF_TYPE_TO_NAMES) {
            uint32_t typeId = static_cast<uint32_t>(iter.first);
            const auto ret = MsprofRegTypeInfo(MSPROF_REPORT_ACL_LEVEL, typeId, iter.second.c_str());
            if (ret != MSPROF_ERROR_NONE) {
                ACL_LOG_CALL_ERROR("Registered api type [%u] failed = %d", typeId, ret);
                return ACL_ERROR_PROFILING_FAILURE;
            }
        }
        return ACL_SUCCESS;
    }

    static aclError AddDeviceList(const uint32_t *const deviceIdList, const uint32_t deviceNums)
    {
        ACL_REQUIRES_NOT_NULL(deviceIdList);
        for (size_t devId = 0U; devId < deviceNums; devId++) {
            if (g_deviceList.count(*(deviceIdList + devId)) == 0U) {
                (void)g_deviceList.insert(*(deviceIdList + devId));
                ACL_LOG_INFO("device id %u is successfully added in acl profiling", *(deviceIdList + devId));
            }
        }
        return ACL_SUCCESS;
    }

    static aclError RemoveDeviceList(const uint32_t *const deviceIdList, const uint32_t deviceNums)
    {
        ACL_REQUIRES_NOT_NULL(deviceIdList);
        for (size_t devId = 0U; devId < deviceNums; devId++) {
            const auto iter = g_deviceList.find(*(deviceIdList + devId));
            if (iter != g_deviceList.end()) {
                (void)g_deviceList.erase(iter);
                ACL_LOG_INFO("device id %u is successfully deleted from acl profiling", *(deviceIdList + devId));
            }
        }
        return ACL_SUCCESS;
    }

    static aclError ProfInnerStart(const MsprofCommandHandle *const profilerConfig)
    {
        ACL_LOG_INFO("start to execute ProfInnerStart");
        if (!g_profRun) {
            RegisterProfType();
            g_profRun = true;
        }
        (void)AddDeviceList(profilerConfig->devIdList, profilerConfig->devNums);
        ACL_LOG_INFO("successfully execute ProfInnerStart");
        return ACL_SUCCESS;
    }

    static aclError ProfInnerStop(const MsprofCommandHandle *const profilerConfig)
    {
        ACL_LOG_INFO("start to execute ProfInnerStop");
        (void)RemoveDeviceList(profilerConfig->devIdList, profilerConfig->devNums);

        if (g_deviceList.empty() && g_profRun) {
            g_profRun = false;
        }
        ACL_LOG_INFO("successfully execute ProfInnerStop");
        return ACL_SUCCESS;
    }

    static aclError ProcessProfData(void *const data, const uint32_t len)
    {
        ACL_LOG_INFO("start to execute ProcessProfData");
       const std::lock_guard<std::mutex> lk(g_profMutex);
        ACL_REQUIRES_NOT_NULL(data);
        constexpr size_t commandLen = sizeof(MsprofCommandHandle);
        if (len < commandLen) {
            ACL_LOG_INNER_ERROR("[Check][Len]len[%u] is invalid, it should not be smaller than %zu", len, commandLen);
            return ACL_ERROR_INVALID_PARAM;
        }
        MsprofCommandHandle *const profilerConfig = static_cast<MsprofCommandHandle *>(data);
        aclError ret = ACL_SUCCESS;
        const uint64_t profSwitch = profilerConfig->profSwitch;
        const uint32_t type = profilerConfig->type;
        if (((profSwitch & ACL_PROF_ACL_API) != 0U) && (type == START_PROFILING)) {
            ret = ProfInnerStart(profilerConfig);
        }
        if (((profSwitch & ACL_PROF_ACL_API) != 0U) && (type == STOP_PROFILING)) {
            ret = ProfInnerStop(profilerConfig);
        }

        return ret;
    }

    static aclError AclProfCtrlHandle(uint32_t dataType, void *data, uint32_t dataLen)
    {
        ACL_REQUIRES_NOT_NULL(data);

        if (dataType == PROF_CTRL_SWITCH) {
            const aclError ret = ProcessProfData(data, dataLen);
            if (ret != ACL_SUCCESS) {
                ACL_LOG_INNER_ERROR("[Process][ProfSwitch]failed to call ProcessProfData, result is %u", ret);
                return ret;
            }
            return ACL_SUCCESS;
        }
 
        ACL_LOG_INFO("get unsupported dataType %u while processing profiling data", dataType);
        return ACL_SUCCESS;
    }


    class AclRegProfCallback {
    public:
        AclRegProfCallback() {
            const auto profRet = MsprofRegisterCallback(ASCENDCL, &AclProfCtrlHandle);
            if (profRet != 0) {
                ACL_LOG_ERROR("can not register Callback, prof result = %d", profRet);
            }
        }
        ~AclRegProfCallback() {}
    };
    static AclRegProfCallback g_profCbReg;
}
namespace acl {

    AclProfilingReporter::AclProfilingReporter(const AclProfType apiId) : aclApi_(apiId)
    {
        if (g_profRun && (!IsDumpToStdEnabled())) {
            startTime_ = MsprofSysCycleTime();
        }
    }


    AclProfilingReporter::~AclProfilingReporter() noexcept
    {
        if (g_profRun && (!IsDumpToStdEnabled()) && (startTime_ != 0UL)) {
            // 1000 ^ 3 converts second to nanosecond
            const uint64_t endTime = MsprofSysCycleTime();
            MsprofApi api{};
            api.beginTime = startTime_;
            api.endTime = endTime;
            thread_local static auto tid = mmGetTid();
            api.threadId = static_cast<uint32_t>(tid);
            api.level = MSPROF_REPORT_ACL_LEVEL;
            api.type = static_cast<uint32_t>(aclApi_);
            (void)MsprofReportApi(true, &api);
        }
    }

}  // namespace acl
