/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_config.h"
#include "common/ge_inner_error_codes.h"
#include "common/log_inner.h"
#include "model_desc_internal.h"
#include "error_codes_inner.h"
#include "acl_model_impl.h"

namespace {
    using CheckMdlConfigFunc = aclError (*)(const void *const, const size_t);
    using SetMdlConfigFunc = aclError (*)(aclmdlConfigHandle *const, const void *const);

    struct SetMdlConfigParamFunc {
        CheckMdlConfigFunc checkFunc;
        SetMdlConfigFunc setFunc;
    };

    aclError CheckMdlLoadPriority(const void *const attrValue, const size_t valueSize)
    {
        ACL_LOG_INFO("start to execute CheckMdlLoadPriority.");
        if (valueSize != sizeof(int32_t)) {
            ACL_LOG_ERROR("[Check][File]valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(int32_t));
            const std::string errMsg = "it should be " + std::to_string(sizeof(int32_t));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }

        static bool mdlLoadPriorityFlag = false;
        static int32_t leastPriority;
        static int32_t greatestPriority;
        if (!mdlLoadPriorityFlag) {
            const aclError aclErr = aclrtDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
            if (aclErr != ACL_ERROR_NONE) {
                ACL_LOG_CALL_ERROR("[Get][PriorityRange]get range of stream priority failed, "
                    "runtime errorCode = %d", static_cast<int32_t>(aclErr));
                return aclErr;
            }
            mdlLoadPriorityFlag = true;
            ACL_LOG_INFO("execute aclrtDeviceGetStreamPriorityRange success, greatest load priority = %d, "
                "least load priority = %d", greatestPriority, leastPriority);
        }
        const int32_t priority = *static_cast<const int32_t *>(attrValue);
        ACL_CHECK_RANGE_INT(priority, greatestPriority, leastPriority);
        ACL_LOG_INFO("successfully execute CheckMdlLoadPriority");
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadPriority(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadPriority.");
        const int32_t priority = *static_cast<const int32_t *>(attrValue);
        handle->priority = priority;
        ACL_LOG_INFO("successfully execute SetMdlLoadPriority");
        return ACL_SUCCESS;
    }

    aclError CheckMdlLoadWithoutGraph(const void *const attrValue, const size_t valueSize)
    {
      ACL_LOG_INFO("start to execute CheckMdlLoadWithoutGraph.");
      if (valueSize != sizeof(int32_t)) {
        ACL_LOG_ERROR("valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(int32_t));
        const std::string errMsg = "it should be " + std::to_string(sizeof(int32_t));
        acl::AclErrorLogManager::ReportInputError(
            acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
        return ACL_ERROR_INVALID_PARAM;
      }
      const int32_t flag = *static_cast<const int32_t *>(attrValue);
      if ((flag != 0) && (flag != 1)) {
        ACL_LOG_ERROR("ACL_MDL_WITHOUT_GRAPH_INT32 value [%d] is invalid, it should be in [0, 1]", flag);
        const std::string errMsg = "it should be in [0, 1]";
        acl::AclErrorLogManager::ReportInputError(
            acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"ACL_MDL_WITHOUT_GRAPH_INT32", std::to_string(flag).c_str(), errMsg.c_str()}));
        return ACL_ERROR_INVALID_PARAM;
      }
      ACL_LOG_INFO("successfully execute CheckMdlLoadPriority");
      return ACL_SUCCESS;
    }

    aclError SetMdlLoadWithoutGraph(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
      ACL_LOG_INFO("start to execute SetMdlLoadWithoutGraph.");
      const int32_t flag = *static_cast<const int32_t *>(attrValue);
      handle->withoutGraph = (flag != 0);
      (void)handle->attrState.insert(ACL_MDL_WITHOUT_GRAPH_INT32);
      ACL_LOG_INFO("successfully execute SetMdlLoadWithoutGraph, flag is %d", flag);
      return ACL_SUCCESS;
    }

    aclError CheckMdlLoadType(const void *const attrValue, const size_t valueSize)
    {
        ACL_LOG_INFO("start to execute CheckMdlLoadType.");
        if (valueSize != sizeof(size_t)) {
            ACL_LOG_ERROR("[Check][ValueSize]valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(size_t));
            const std::string errMsg = "it should be " + std::to_string(sizeof(size_t));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        const size_t type = *static_cast<const size_t *>(attrValue);
        if ((type < static_cast<size_t>(ACL_MDL_LOAD_FROM_FILE)) ||
            (type > static_cast<size_t>(ACL_MDL_LOAD_FROM_MEM_WITH_Q))) {
            ACL_LOG_ERROR("[Check][Type]type[%zu] is invalid, it should be in [%d, %d]",
                type, ACL_MDL_LOAD_FROM_FILE, ACL_MDL_LOAD_FROM_MEM_WITH_Q);
            const std::string errMsg = acl::AclErrorLogManager::FormatStr("it should be in [%d, %d]",
                ACL_MDL_LOAD_FROM_FILE, ACL_MDL_LOAD_FROM_MEM_WITH_Q);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"*attrValue", std::to_string(type).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        ACL_LOG_INFO("successfully execute CheckMdlLoadType to check aclmdlLoadType[%zu]", type);
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadType(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadType.");
        const size_t type = *static_cast<const size_t *>(attrValue);
        handle->mdlLoadType = type;
        (void)handle->attrState.insert(ACL_MDL_LOAD_TYPE_SIZET);
        ACL_LOG_INFO("successfully execute SetMdlLoadType to set aclmdlLoadType[%zu]", type);
        return ACL_SUCCESS;
    }

    // compared with CheckMdlLoadPtrAttr, the CheckMdlLoadPtrAttrEx requires the values attrValue point to
    // can't be nullptr
    aclError CheckMdlLoadPtrAttrEx(const void *const attrValue, const size_t valueSize)
    {
        ACL_LOG_INFO("start to execute CheckMdlLoadPtrAttrEx.");
        if (valueSize != sizeof(void *)) {
            ACL_LOG_ERROR("[Check][ValueSize]valueSize[%zu] is invalid, it should be %zu",
                valueSize, sizeof(void *));
            const std::string errMsg = "it should be " + std::to_string(sizeof(void *));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        const void *const val = *static_cast<void * const *>(attrValue);
        ACL_REQUIRES_NOT_NULL(val);
        ACL_LOG_INFO("successfully execute CheckMdlLoadPtrAttrEx");
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadPath(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadPath");
        // set model path with deep copy
        const char_t *const loadPath = *static_cast<char_t * const *>(attrValue);
        handle->loadPath = loadPath;
        (void)handle->attrState.insert(ACL_MDL_PATH_PTR);
        ACL_LOG_INFO("successfully execute SetMdlLoadPath to set loadPath[%s]", loadPath);
        return ACL_SUCCESS;
    }

    aclError SetMdlWeightPath(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlWeightPath");
        // set model path with deep copy
        char_t *const weightPath = *static_cast<char_t * const *>(attrValue);
        handle->weightPath = weightPath;
        (void)handle->attrState.insert(ACL_MDL_WEIGHT_PATH_PTR);
        ACL_LOG_INFO("successfully execute SetMdlWeightPath to set weightPath[%s]", weightPath);
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadMemPtr(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadMemPtr");
        void *const val = *static_cast<void * const *>(attrValue);
        handle->mdlAddr = val;
        (void)handle->attrState.insert(ACL_MDL_MEM_ADDR_PTR);
        ACL_LOG_INFO("successfully execute SetMdlLoadMemPtr");
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadInputQPtr(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadInputQPtr");
        void *const val = *static_cast<void * const *>(attrValue);
        handle->inputQ = static_cast<uint32_t *>(val);
        (void)handle->attrState.insert(ACL_MDL_INPUTQ_ADDR_PTR);
        ACL_LOG_INFO("successfully execute SetMdlLoadInputQPtr");
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadOutputQPtr(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadOutputQPtr");
        void *const val = *static_cast<void * const *>(attrValue);
        handle->outputQ = static_cast<uint32_t *>(val);
        (void)handle->attrState.insert(ACL_MDL_OUTPUTQ_ADDR_PTR);
        ACL_LOG_INFO("successfully execute SetMdlLoadOutputQPtr");
        return ACL_SUCCESS;
    }

    aclError CheckMdlLoadSizeAttr(const void *const attrValue, const size_t valueSize)
    {
        (void)attrValue;
        ACL_LOG_INFO("start to execute CheckMdlLoadSizeAttr");
        if (valueSize != sizeof(size_t)) {
            ACL_LOG_ERROR("[Check][ValueSize]valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(size_t));
            const std::string errMsg = "it should be " + std::to_string(sizeof(size_t));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        ACL_LOG_INFO("successfully execute CheckMdlLoadSizeAttr");
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadWeightSize(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadWeightSize");
        const size_t weightSize = *static_cast<const size_t *>(attrValue);
        handle->weightSize = weightSize;
        ACL_LOG_INFO("successfully execute SetMdlLoadWeightSize to set weightSize[%zu]", weightSize);
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadWorkspaceSize(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadWorkspaceSize");
        const size_t workSize = *static_cast<const size_t *>(attrValue);
        handle->workSize = workSize;
        ACL_LOG_INFO("successfully execute SetMdlLoadWorkspaceSize to set workSize[%zu]", workSize);
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadInputQNum(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadInputQNum");
        const size_t num = *static_cast<const size_t *>(attrValue);
        handle->inputQNum = num;
        (void)handle->attrState.insert(ACL_MDL_INPUTQ_NUM_SIZET);
        ACL_LOG_INFO("successfully execute SetMdlLoadInputQNum to set inputQNum[%zu]", num);
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadOutputQNum(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadOutputQNum");
        const size_t num = *static_cast<const size_t *>(attrValue);
        handle->outputQNum = num;
        (void)handle->attrState.insert(ACL_MDL_OUTPUTQ_NUM_SIZET);
        ACL_LOG_INFO("successfully execute SetMdlLoadOutputQNum to set outputQNum[%zu]", num);
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadMemSize(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadMemSize");
        const size_t memSize = *static_cast<const size_t *>(attrValue);
        handle->mdlSize = memSize;
        (void)handle->attrState.insert(ACL_MDL_MEM_SIZET);
        ACL_LOG_INFO("successfully execute SetMdlLoadMemSize to set memSize[%zu]", memSize);
        return ACL_SUCCESS;
    }

    aclError CheckMdlLoadPtrAttr(const void *const attrValue, const size_t valueSize)
    {
        (void)attrValue;
        ACL_LOG_INFO("start to execute CheckMdlLoadPtrAttr");
        if (valueSize != sizeof(void *)) {
            ACL_LOG_ERROR("[Check][ValueSize]valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(void *));
            const std::string errMsg = "it should be " + std::to_string(sizeof(void *));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        ACL_LOG_INFO("successfully execute CheckMdlLoadPtrAttr");
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadWeightPtr(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadWeightPtr");
        void *const val = *static_cast<void * const *>(attrValue);
        handle->weightPtr = val;
        ACL_LOG_INFO("successfully execute SetMdlLoadWeightPtr");
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadWorkPtr(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadWorkPtr");
        void *const val = *static_cast<void * const *>(attrValue);
        handle->workPtr = val;
        ACL_LOG_INFO("successfully execute SetMdlLoadWorkPtr");
        return ACL_SUCCESS;
    }

    aclError CheckMdlLoadReuseZeroCopy(const void *const attrValue, const size_t valueSize)
    {
        ACL_LOG_INFO("start to execute CheckMdlLoadReuseZeroCopy.");
        if (valueSize != sizeof(size_t)) {
            ACL_LOG_ERROR("[Check][ValueSize]valueSize[%zu] is invalid, it should be %zu",
                valueSize, sizeof(size_t));
            const std::string errMsg = "it should be " + std::to_string(sizeof(size_t));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        const size_t type = *static_cast<const size_t *>(attrValue);
        if ((type > static_cast<size_t>(ACL_WORKSPACE_MEM_OPTIMIZE_INPUTOUTPUT))) {
            ACL_LOG_ERROR("[Check][Type]type[%zu] is invalid, it should be in [%d, %d]",
                type, ACL_WORKSPACE_MEM_OPTIMIZE_DEFAULT, ACL_WORKSPACE_MEM_OPTIMIZE_INPUTOUTPUT);
            const std::string errMsg = acl::AclErrorLogManager::FormatStr("it should be in [%d, %d]",
                ACL_WORKSPACE_MEM_OPTIMIZE_DEFAULT, ACL_WORKSPACE_MEM_OPTIMIZE_INPUTOUTPUT);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"*attrValue", std::to_string(type).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        ACL_LOG_INFO("successfully execute CheckMdlLoadReuseZeroCopy to check optimize type[%zu]", type);
        return ACL_SUCCESS;
    }

    aclError SetMdlLoadReuseZeroCopy(aclmdlConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlLoadReuseZeroCopy");
        const size_t val = *static_cast<const size_t *>(attrValue);
        handle->reuseZeroCopy = val;
        ACL_LOG_INFO("successfully execute SetMdlLoadReuseZeroCopy");
        return ACL_SUCCESS;
    }

    std::map<aclmdlConfigAttr, SetMdlConfigParamFunc> g_setMdlConfigMap = {
        {ACL_MDL_PRIORITY_INT32, {&CheckMdlLoadPriority, &SetMdlLoadPriority}},
        {ACL_MDL_LOAD_TYPE_SIZET, {&CheckMdlLoadType, &SetMdlLoadType}},
        {ACL_MDL_PATH_PTR, {&CheckMdlLoadPtrAttrEx, &SetMdlLoadPath}},
        {ACL_MDL_MEM_ADDR_PTR, {&CheckMdlLoadPtrAttrEx, &SetMdlLoadMemPtr}},
        {ACL_MDL_INPUTQ_ADDR_PTR, {&CheckMdlLoadPtrAttrEx, &SetMdlLoadInputQPtr}},
        {ACL_MDL_OUTPUTQ_ADDR_PTR, {&CheckMdlLoadPtrAttrEx, &SetMdlLoadOutputQPtr}},
        {ACL_MDL_MEM_SIZET, {&CheckMdlLoadSizeAttr, &SetMdlLoadMemSize}},
        {ACL_MDL_WEIGHT_SIZET, {&CheckMdlLoadSizeAttr, &SetMdlLoadWeightSize}},
        {ACL_MDL_WORKSPACE_SIZET, {&CheckMdlLoadSizeAttr, &SetMdlLoadWorkspaceSize}},
        {ACL_MDL_INPUTQ_NUM_SIZET, {&CheckMdlLoadSizeAttr, &SetMdlLoadInputQNum}},
        {ACL_MDL_OUTPUTQ_NUM_SIZET, {&CheckMdlLoadSizeAttr, &SetMdlLoadOutputQNum}},
        {ACL_MDL_WEIGHT_ADDR_PTR, {&CheckMdlLoadPtrAttr, &SetMdlLoadWeightPtr}},
        {ACL_MDL_WORKSPACE_ADDR_PTR, {&CheckMdlLoadPtrAttr, &SetMdlLoadWorkPtr}},
        {ACL_MDL_WORKSPACE_MEM_OPTIMIZE, {&CheckMdlLoadReuseZeroCopy, &SetMdlLoadReuseZeroCopy}},
        {ACL_MDL_WEIGHT_PATH_PTR, {&CheckMdlLoadPtrAttrEx, &SetMdlWeightPath}},
        {ACL_MDL_WITHOUT_GRAPH_INT32, {&CheckMdlLoadWithoutGraph, &SetMdlLoadWithoutGraph}}
    };

    using CheckMdlExecConfigFunc = aclError (*)(const void *const, const size_t);
    using SetMdlExecConfigFunc = aclError (*)(aclmdlExecConfigHandle *const, const void *const);

    struct SetMdlExecConfigParamFunc {
        CheckMdlExecConfigFunc checkFunc;
        SetMdlExecConfigFunc setFunc;
    };

    aclError CheckMdlStreamSyncTimeout(const void *const attrValue, const size_t valueSize)
    {
        ACL_LOG_INFO("start to execute CheckMdlStreamSyncTimeout.");
        if (valueSize != sizeof(int32_t)) {
            ACL_LOG_ERROR("[Check][File]valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(int32_t));
            const std::string errMsg = "it should be " + std::to_string(sizeof(int32_t));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        const int32_t streamSyncTimeout = *static_cast<const int32_t *>(attrValue);
        constexpr int32_t leastTimeout = -1;
        constexpr int32_t greatestTimeout = INT32_MAX;
        ACL_CHECK_RANGE_INT(streamSyncTimeout, leastTimeout, greatestTimeout);
        ACL_LOG_INFO("successfully execute CheckMdlStreamSyncTimeout");
        return ACL_SUCCESS;
    }

    aclError SetMdlStreamSyncTimeout(aclmdlExecConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlStreamSyncTimeout.");
        const int32_t streamSyncTimeout = *static_cast<const int32_t *>(attrValue);
        handle->streamSyncTimeout = streamSyncTimeout;
        ACL_LOG_INFO("successfully execute SetMdlStreamSyncTimeout");
        return ACL_SUCCESS;
    }

    aclError CheckMdlEventSyncTimeout(const void *const attrValue, const size_t valueSize)
    {
        ACL_LOG_INFO("start to execute CheckMdlEventSyncTimeout.");
        if (valueSize != sizeof(int32_t)) {
            ACL_LOG_ERROR("[Check][File]valueSize[%zu] is invalid, it should be %zu", valueSize, sizeof(int32_t));
            const std::string errMsg = "it should be " + std::to_string(sizeof(int32_t));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"valueSize", std::to_string(valueSize).c_str(), errMsg.c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        const int32_t eventSyncTimeout = *static_cast<const int32_t *>(attrValue);
        constexpr int32_t leastTimeout = -1;
        constexpr int32_t greatestTimeout = INT32_MAX;
        ACL_CHECK_RANGE_INT(eventSyncTimeout, leastTimeout, greatestTimeout);
        ACL_LOG_INFO("successfully execute CheckMdlEventSyncTimeout");
        return ACL_SUCCESS;
    }

    aclError SetMdlEventSyncTimeout(aclmdlExecConfigHandle *const handle, const void *const attrValue)
    {
        ACL_LOG_INFO("start to execute SetMdlEventSyncTimeout.");
        const int32_t eventSyncTimeout = *static_cast<const int32_t *>(attrValue);
        handle->eventSyncTimeout = eventSyncTimeout;
        ACL_LOG_INFO("successfully execute SetMdlEventSyncTimeout");
        return ACL_SUCCESS;
    }

    std::map<aclmdlExecConfigAttr, SetMdlExecConfigParamFunc> g_setMdlExecConfigMap = {
            {ACL_MDL_STREAM_SYNC_TIMEOUT, {&CheckMdlStreamSyncTimeout, &SetMdlStreamSyncTimeout}},
            {ACL_MDL_EVENT_SYNC_TIMEOUT, {&CheckMdlEventSyncTimeout, &SetMdlEventSyncTimeout}}
    };
} // anonymous namespace

namespace acl {
static bool CheckMdlLoadConfigFromFile(const aclmdlConfigHandle *const handle)
{
    if (handle->attrState.find(ACL_MDL_PATH_PTR) == handle->attrState.end()) {
        ACL_LOG_ERROR("[Check][Type]model load type[%zu]: model path is not set in aclmdlConfigHandle "
                      "when load type is from file", handle->mdlLoadType);
        const std::string errMsg = "model path is not set in aclmdlConfigHandle when load type is from file";
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"handle", "inner model path", errMsg.c_str()}));
        return false;
    }
    if (handle->attrState.find(ACL_MDL_WEIGHT_PATH_PTR) != handle->attrState.end()) {
        ACL_LOG_ERROR("[Check][Type]model load type[%zu]: should not set ACL_MDL_WEIGHT_PATH_PTR",
            handle->mdlLoadType);
        const std::string errMsg = "should not set ACL_MDL_WEIGHT_PATH_PTR";
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"handle", "weight path", errMsg.c_str()}));
        return false;
    }
    return true;
}

static bool CheckMdlLoadConfigFromMem(const aclmdlConfigHandle *const handle)
{
    if (handle->attrState.find(ACL_MDL_MEM_ADDR_PTR) == handle->attrState.end()) {
        ACL_LOG_ERROR("[Check][Type]model load type[%zu]: model memory ptr is not set in aclmdlConfigHandle",
            handle->mdlLoadType);
        const std::string errMsg = "model memory ptr is not set in aclmdlConfigHandle when load type is from mem";
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"handle", "inner model memory", errMsg.c_str()}));
        return false;
    }

    if (handle->attrState.find(ACL_MDL_MEM_SIZET) == handle->attrState.end()) {
        ACL_LOG_ERROR("[Check][Type]model load type[%zu]: model memory size is not set in aclmdlConfigHandle",
            handle->mdlLoadType);
        const std::string errMsg = "model memory size is not set in aclmdlConfigHandle when load type is from mem";
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"handle", "inner model memory size", errMsg.c_str()}));
        return false;
    }
    if ((handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_MEM_WITH_MEM)) &&
            (handle->attrState.find(ACL_MDL_WEIGHT_PATH_PTR) != handle->attrState.end())) {
        ACL_LOG_ERROR("[Check][Type]model load type[%zu]: should not set ACL_MDL_WEIGHT_PATH_PTR",
            handle->mdlLoadType);
        const std::string errMsg = "should not set ACL_MDL_WEIGHT_PATH_PTR in aclmdlConfigHandle "
                                   "when load type is ACL_MDL_LOAD_FROM_MEM_WITH_MEM";
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"handle", "inner weight path", errMsg.c_str()}));
        return false;
    }
    return true;
}

static bool CheckMdlLoadConfigWithQ(const aclmdlConfigHandle *const handle)
{
        if (handle->attrState.find(ACL_MDL_INPUTQ_ADDR_PTR) == handle->attrState.end()) {
            ACL_LOG_ERROR("[Check][Type]model load type[%zu]: inputQ ptr is not set in aclmdlConfigHandle",
                handle->mdlLoadType);
            const std::string errMsg = "ACL_MDL_INPUTQ_ADDR_PTR is not set in aclmdlConfigHandle "
                                       "when load type is with queue";
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"handle", "inner inputq addr", errMsg.c_str()}));
            return false;
        }

        if (handle->attrState.find(ACL_MDL_INPUTQ_NUM_SIZET) == handle->attrState.end()) {
            ACL_LOG_ERROR("[Check][Type]model load type[%zu]: inputQ num is not set in aclmdlConfigHandle",
                handle->mdlLoadType);
            const std::string errMsg = "ACL_MDL_INPUTQ_NUM_SIZET is not set in aclmdlConfigHandle "
                                       "when load type is with queue";
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"handle", "inner inputq num", errMsg.c_str()}));
            return false;
        }

        if (handle->attrState.find(ACL_MDL_OUTPUTQ_ADDR_PTR) == handle->attrState.end()) {
            ACL_LOG_ERROR("[Check][Type]model load type[%zu]: outputQ ptr is not set in aclmdlConfigHandle",
                handle->mdlLoadType);
            const std::string errMsg = "ACL_MDL_OUTPUTQ_ADDR_PTR is not set in aclmdlConfigHandle "
                                       "when load type is with queue";
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"handle", "inner outputq addr", errMsg.c_str()}));
            return false;
        }

        if (handle->attrState.find(ACL_MDL_OUTPUTQ_NUM_SIZET) == handle->attrState.end()) {
            ACL_LOG_ERROR("[Check][Type]model load type[%zu]: outputQ num is not set in aclmdlConfigHandle",
                handle->mdlLoadType);
            const std::string errMsg = "ACL_MDL_OUTPUTQ_NUM_SIZET is not set in aclmdlConfigHandle "
                                       "when load type is with queue";
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"handle", "inner outputq num", errMsg.c_str()}));
            return false;
        }

        return true;
}

bool CheckMdlConfigHandle(const aclmdlConfigHandle *const handle)
{
    if (handle->attrState.find(ACL_MDL_LOAD_TYPE_SIZET) == handle->attrState.end()) {
        ACL_LOG_ERROR("[Find][Type]model load type is not set in aclmdlConfigHandle");
        const std::string errMsg = "ACL_MDL_LOAD_TYPE_SIZET is not set in aclmdlConfigHandle";
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"handle", "inner load type", errMsg.c_str()}));
        return false;
    }

    if ((handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_FILE)) ||
        (handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_FILE_WITH_MEM))) {
        if (!CheckMdlLoadConfigFromFile(handle)) {
            return false;
        }
    }

    if ((handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_MEM)) ||
        (handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_MEM_WITH_MEM))) {
        if ((!CheckMdlLoadConfigFromMem(handle))) {
            return false;
        }
    }

    if (handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_FILE_WITH_Q)) {
        if ((!CheckMdlLoadConfigFromFile(handle)) || (!CheckMdlLoadConfigWithQ(handle))) {
            return false;
        }
    }

    if (handle->mdlLoadType == static_cast<size_t>(ACL_MDL_LOAD_FROM_MEM_WITH_Q)) {
        if ((!CheckMdlLoadConfigFromMem(handle)) || (!CheckMdlLoadConfigWithQ(handle))) {
            return false;
        }
    }
    return true;
}
}

aclError aclmdlSetConfigOptImpl(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
    const void *attrValue, size_t valueSize)
{
    ACL_LOG_INFO("start to execute aclmdlSetConfigOpt, attr[%d]", static_cast<int32_t>(attr));
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrValue);
    SetMdlConfigParamFunc paramFunc;
    if (g_setMdlConfigMap.find(attr) != g_setMdlConfigMap.end()) {
        paramFunc = g_setMdlConfigMap[attr];
    } else {
        ACL_LOG_ERROR("[Find][Attr]attr[%d] is invalid. it should be [%d, %d]", static_cast<int32_t>(attr),
            static_cast<int32_t>(ACL_MDL_PRIORITY_INT32), static_cast<int32_t>(ACL_MDL_WEIGHT_PATH_PTR));
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("it should be [%d, %d]",
            static_cast<int32_t>(ACL_MDL_PRIORITY_INT32), static_cast<int32_t>(ACL_MDL_WEIGHT_PATH_PTR));
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"attr", std::to_string(static_cast<int32_t>(attr)).c_str(), errMsg.c_str()}));
        return ACL_ERROR_INVALID_PARAM;
    }
    aclError ret = paramFunc.checkFunc(attrValue, valueSize);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("[Check][Params]check params by aclmdlSetConfigOpt error, result[%d], attr[%d], "
            "valueSize[%zu]", ret, static_cast<int32_t>(attr), valueSize);
        return ret;
    }
    ret = paramFunc.setFunc(handle, attrValue);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("[Set][Params]set params by aclmdlSetConfigOpt error, result[%d], attr[%d]", ret,
            static_cast<int32_t>(attr));
        return ret;
    }

    ACL_LOG_INFO("successfully execute aclmdlSetConfigOpt, attr[%d]", static_cast<int32_t>(attr));
    return ACL_SUCCESS;
}

aclError aclmdlSetExternalWeightAddressImpl(aclmdlConfigHandle *handle, const char *weightFileName,
    void *devPtr, size_t size) {
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(weightFileName);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(devPtr);
    if ((size == 0U) || (size % 32U != 0U)) {
        ACL_LOG_ERROR("[Check][Params]memSize[%zu] must > 0 and 32-byte aligned.", size);
        return ACL_ERROR_INVALID_PARAM;
    }

    ge::FileConstantMem fileConstnatMem{.file_name = weightFileName, .device_mem = devPtr, .mem_size = size};
    handle->fileConstantMem.emplace_back(std::move(fileConstnatMem));
    ACL_LOG_INFO("successfully to execute aclmdlSetExternalWeightAddress, weightFileName[%s], devPtr[%p], size[%zu]",
                 weightFileName, devPtr, size);
    return ACL_SUCCESS;
}

aclmdlConfigHandle *aclmdlCreateConfigHandleImpl()
{
    return new(std::nothrow) aclmdlConfigHandle();
}

aclError aclmdlDestroyConfigHandleImpl(aclmdlConfigHandle *handle)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    ACL_DELETE_AND_SET_NULL(handle);
    return ACL_SUCCESS;
}

aclError aclmdlSetExecConfigOptImpl(aclmdlExecConfigHandle *handle, aclmdlExecConfigAttr attr,
                                const void *attrValue, size_t valueSize)
{
    ACL_LOG_INFO("start to execute aclmdlSetExecConfigOpt, attr[%d]", static_cast<int32_t>(attr));
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(attrValue);
    SetMdlExecConfigParamFunc paramFunc;
    if (g_setMdlExecConfigMap.find(attr) != g_setMdlExecConfigMap.end()) {
        paramFunc = g_setMdlExecConfigMap[attr];
    } else {
        ACL_LOG_ERROR("[Find][Attr]attr[%d] is invalid. it should be [%d, %d]", static_cast<int32_t>(attr),
                            static_cast<int32_t>(ACL_MDL_STREAM_SYNC_TIMEOUT),
                            static_cast<int32_t>(ACL_MDL_EVENT_SYNC_TIMEOUT));
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("it should be [%d, %d]",
            static_cast<int32_t>(ACL_MDL_STREAM_SYNC_TIMEOUT), static_cast<int32_t>(ACL_MDL_EVENT_SYNC_TIMEOUT));
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"attr", std::to_string(static_cast<int32_t>(attr)).c_str(), errMsg.c_str()}));
        return ACL_ERROR_INVALID_PARAM;
    }
    aclError ret = paramFunc.checkFunc(attrValue, valueSize);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("[Check][Params]check params by aclmdlSetExecConfigOpt error, result[%d], attr[%d], "
                            "valueSize[%zu]", ret, static_cast<int32_t>(attr), valueSize);
        return ret;
    }
    ret = paramFunc.setFunc(handle, attrValue);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("[Set][Params]set params by aclmdlSetExecConfigOpt error, result[%d], attr[%d]", ret,
                            static_cast<int32_t>(attr));
        return ret;
    }

    ACL_LOG_INFO("successfully execute aclmdlSetExecConfigOpt, attr[%d]", static_cast<int32_t>(attr));
    return ACL_SUCCESS;
}
