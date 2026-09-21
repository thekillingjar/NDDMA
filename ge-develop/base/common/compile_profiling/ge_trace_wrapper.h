/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_TRACE_WRAPPER_H_
#define GE_GE_TRACE_WRAPPER_H_

#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/scope_tracing_recorder.h"

#define GE_TRACE_START(stage) \
  const uint64_t startUsec_##stage = ge::GetCurrentTimestamp(); \
  TRACING_DURATION_START(stage)

#define GE_TRACE_TIMESTAMP_END_INNER(stage, tag, stage_name)                                    \
  do {                                                                               \
    const uint64_t endUsec_##tag = ge::GetCurrentTimestamp();                        \
    TRACING_DURATION_END(stage, tag, stage_name);                                    \
    GELOGI("[GEPERFTRACE] The time cost of %s is [%lu] micro seconds.", (stage_name),\
            (endUsec_##tag - startUsec_##tag));                                      \
  } while (false)

#define GE_INIT_TRACE_TIMESTAMP_END(stage, stage_name) \
    GE_TRACE_TIMESTAMP_END_INNER(ge::TracingModule::kCANNInitialize, stage, stage_name)

#define GE_LOAD_TRACE_TIMESTAMP_END(stage, stage_name) \
    GE_TRACE_TIMESTAMP_END_INNER(ge::TracingModule::kModelLoad, stage, stage_name)

#define GE_COMPILE_TRACE_TIMESTAMP_END(stage, stage_name) \
    GE_TRACE_TIMESTAMP_END_INNER(ge::TracingModule::kModelCompile, stage, stage_name)

#define RETURN_IF_ERROR_WITH_TRACE_TIMESTAMP_NAME(var_name, prefix, func, ...) \
  do {                                                                         \
    GE_TRACE_START(var_name);                                                  \
    const auto ret_inner_macro = (func)(__VA_ARGS__);                          \
    GE_COMPILE_TRACE_TIMESTAMP_END(var_name, #prefix "::" #func);              \
    if (ret_inner_macro != ge::SUCCESS) {                                     \
      GELOGE(ret_inner_macro, "[Process][" #prefix "_" #func "] failed");     \
      return ret_inner_macro;                                                 \
    }                                                                         \
  } while (false)

#define GE_TRACE_RUN(prefix, func, ...) \
  RETURN_IF_ERROR_WITH_TRACE_TIMESTAMP_NAME(COUNTER_NAME(ge_timestamp_##prefix), prefix, func, __VA_ARGS__)
#define GE_TRACE_RUN_PERF(prefix, func, ...) \
  RETURN_IF_ERROR_WITH_TRACE_TIMESTAMP_NAME(COUNTER_NAME(ge_timestamp_##prefix), prefix, func, __VA_ARGS__)

#endif  // GE_GE_CALL_WRAPPER_H_
