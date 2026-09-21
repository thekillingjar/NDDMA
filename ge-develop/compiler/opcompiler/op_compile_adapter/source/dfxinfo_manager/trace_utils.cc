/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfxinfo_manager/trace_utils.h"
#include <sstream>
#include "common/common_utils.h"
#include "atrace_pub.h"

namespace te {
namespace fusion {
namespace {
constexpr char const *kGlobalTraceHandleName = "FE_Global_Trace";
constexpr char const *kCompileTraceHandlePrefix = "FE_CompileTd_";
const uint64_t kTreadIdReminder = 10;
}
void TraceUtils::SubmitGlobalTrace(const std::string &traceMsg)
{
    if (traceMsg.empty()) {
        return;
    }
    auto traceHandle = AtraceGetHandle(TracerType::TRACER_TYPE_SCHEDULE, kGlobalTraceHandleName);
    (void)SubmitTraceInfo(traceHandle, traceMsg);
}

void TraceUtils::SubmitCompileDetailTrace(const uint64_t threadId, const int64_t opId, const std::string &opType,
                                          const std::string &action)
{
    std::stringstream ss;
    ss << "Compile process detail:" << "Thread Id:" << std::to_string(threadId);
    ss << "|Op Id:" << std::to_string(opId) << "|Op Type:" << opType;
    ss << "|" << action;
    SubmitCompileDetailTrace(threadId, ss.str());
}

void TraceUtils::SubmitCompileDetailTrace(const uint64_t threadId, const std::string &traceMsg)
{
    if (traceMsg.empty()) {
        return;
    }
    const std::string traceName = kCompileTraceHandlePrefix + std::to_string((threadId % kTreadIdReminder));
    auto traceHandle = AtraceGetHandle(TracerType::TRACER_TYPE_SCHEDULE, traceName.c_str());
    (void)SubmitTraceInfo(traceHandle, traceMsg);
}

bool TraceUtils::SubmitTraceInfo(const TraHandle &traceHandle, const std::string &traceMsg)
{
    if (traceMsg.empty() || traceHandle < 0) {
        return true;
    }
    uint32_t msgSize = static_cast<uint32_t>(traceMsg.size());
    const void *buffer = reinterpret_cast<const void *>(traceMsg.data());
    uint32_t bufSize = msgSize > DEFAULT_ATRACE_MSG_SIZE ? DEFAULT_ATRACE_MSG_SIZE : msgSize;
    TraStatus trace_status = AtraceSubmit(traceHandle, buffer, bufSize);
    return trace_status == TRACE_SUCCESS;
}
}
}
