/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_CALL_WRAPPER_H_
#define GE_GE_CALL_WRAPPER_H_

#include <pthread.h>

#include "framework/common/debug/ge_log.h"
#include "common/util.h"

/*lint --emacro((773),GE_TIMESTAMP_START)*/
/*lint -esym(773,GE_TIMESTAMP_START)*/
#define GE_TIMESTAMP_START(stage) const uint64_t startUsec_##stage = ge::GetCurrentTimestamp()

#define GE_TIMESTAMP_END(stage, stage_name)                                           \
  do {                                                                                \
    const uint64_t endUsec_##stage = ge::GetCurrentTimestamp();                       \
    GELOGI("[GEPERFTRACE] The time cost of %s is [%lu] micro seconds.", (stage_name),  \
            (endUsec_##stage - startUsec_##stage));                                   \
  } while (false)

#define GE_TIMESTAMP_EVENT_END(stage, stage_name)                                     \
  do {                                                                                \
    const uint64_t endUsec_##stage = ge::GetCurrentTimestamp();                       \
    GEEVENT("[GEPERFTRACE] The time cost of %s is [%lu] micro seconds.", (stage_name), \
            (endUsec_##stage - startUsec_##stage));                                   \
  } while (false)

// flag is used to control whether to print the time cost
#define GE_TIMESTAMP_EVENT_END_WITH_FLAG(stage, stage_name, flag)                       \
  do {                                                                                  \
    if (flag) {                                                                         \
      const uint64_t endUsec_##stage = ge::GetCurrentTimestamp();                       \
      GEEVENT("[GEPERFTRACE] The time cost of %s is [%lu] micro seconds.", (stage_name), \
              (endUsec_##stage - startUsec_##stage));                                   \
    }                                                                                   \
  } while (false)

#define GE_TIMESTAMP_CALLNUM_START(stage)                 \
  uint64_t startUsec_##stage = ge::GetCurrentTimestamp(); \
  uint64_t call_num_of##stage = 0U;                       \
  uint64_t time_of##stage = 0U

#define GE_TIMESTAMP_RESTART(stage) (startUsec_##stage = ge::GetCurrentTimestamp())

#define GE_TIMESTAMP_ADD(stage)                                    \
  time_of##stage += ge::GetCurrentTimestamp() - startUsec_##stage; \
  call_num_of##stage++

#define GE_TIMESTAMP_CALLNUM_END(stage, stage_name)                                                                   \
  GELOGI("[GEPERFTRACE] The time cost of %s is [%lu] micro seconds, call count is %lu", (stage_name), time_of##stage, \
         call_num_of##stage)

#define GE_TIMESTAMP_CALLNUM_EVENT_END(stage, stage_name)                                                           \
  GEEVENT("[GEPERFTRACE] The time cost of %s is [%lu] micro seconds, call count is %lu", (stage_name), time_of##stage, \
          call_num_of##stage)

#define RETURN_IF_ERROR_WITH_TIMESTAMP_NAME(var_name, prefix, func, ...)  \
  do {                                                                    \
    GE_TIMESTAMP_START(var_name);                                         \
    const auto ret_inner_macro = (func)(__VA_ARGS__);                     \
    GE_TIMESTAMP_END(var_name, #prefix "::" #func);                       \
    if (ret_inner_macro != ge::SUCCESS) {                                 \
      GELOGE(ret_inner_macro, "[Process][" #prefix "_" #func "] failed"); \
      return ret_inner_macro;                                             \
    }                                                                     \
  } while (false)

#define RETURN_IF_ERROR_WITH_PERF_TIMESTAMP_NAME(var_name, prefix, func, ...) \
  do {                                                                        \
    GE_TIMESTAMP_START(var_name);                                             \
    const auto ret_inner_macro = (func)(__VA_ARGS__);                         \
    GE_TIMESTAMP_EVENT_END(var_name, #prefix "::" #func);                     \
    if (ret_inner_macro != ge::SUCCESS) {                                     \
      GELOGE(ret_inner_macro, "[Process][" #prefix "_" #func "] failed");     \
      return ret_inner_macro;                                                 \
    }                                                                         \
  } while (false)

#define JOIN_NAME_INNER(a, b) a##b
#define JOIN_NAME(a, b) JOIN_NAME_INNER(a, b)
#define COUNTER_NAME(a) JOIN_NAME(a, __COUNTER__)
#define GE_RUN(prefix, func, ...) \
  RETURN_IF_ERROR_WITH_TIMESTAMP_NAME(COUNTER_NAME(ge_timestamp_##prefix), prefix, func, __VA_ARGS__)
#define GE_RUN_PERF(prefix, func, ...) \
  RETURN_IF_ERROR_WITH_PERF_TIMESTAMP_NAME(COUNTER_NAME(ge_timestamp_##prefix), (prefix), (func), __VA_ARGS__)
#define SET_THREAD_NAME(thread_id, thread_name)       \
  do {                                                \
    (void)pthread_setname_np((thread_id), (thread_name)); \
  } while (false)


namespace ge {
class FuncPerfScope {
 public:
  FuncPerfScope(const char *const message1, const char *const message2)
      : message1_(message1), message2_(message2) {
    start_ = std::chrono::high_resolution_clock::now();
  }
  ~FuncPerfScope() {
    const auto end = std::chrono::high_resolution_clock::now();
    GEEVENT("[GEPERFTRACE] The time cost of %s::%s is [%lu] micro seconds.", message1_, message2_,
            std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count());
  }

 private:
  std::chrono::high_resolution_clock::time_point start_;
  const char *const message1_;
  const char *const message2_;
};
}

#endif  // GE_GE_CALL_WRAPPER_H_
