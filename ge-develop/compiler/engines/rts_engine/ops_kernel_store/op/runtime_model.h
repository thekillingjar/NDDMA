/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CCE_RUNTIME_RT_MODEL_H
#define CCE_RUNTIME_RT_MODEL_H
#if defined(__cplusplus)
extern "C" {
#endif
#define RT_MEMORY_DEFAULT (0x0U)  // default memory on device
#define RT_MEMORY_TS (0x40U)      // Used for Ts memory
typedef enum tagRtCmoType {
  RT_CMO_PREFETCH = 6,  // Preload
  RT_CMO_WRITEBACK,     // Prewriteback
  RT_CMO_INVALID,       // invalid
  RT_CMO_FLUSH,         // flush
  RT_CMO_RESERVED,
} rtCmoOpCode_t;

constexpr uint64_t MAX_MEMCPY_SIZE_OF_D2D = 4ULL * 1024ULL * 1024ULL * 1024ULL;  // 4G
constexpr uint64_t RT_MEMCPYASYNC_SPLIT_SIZE = 67108864UL;                       // 64*1024*1024

#define EXECUTOR_NONE (0x0U)
#define EXECUTOR_TS (0x01U)
#define EXECUTOR_AICPU (0x02U)

/*
 * @ingroup rt_model
 * @brief debug flag for kernel exception dump
 */
#define RT_DEBUG_FLAG_AICORE_OVERFLOW (0x1U << 0U)
#define RT_DEBUG_FLAG_ATOMIC_ADD_OVERFLOW (0x1U << 1U)

#if defined(__cplusplus)
}
#endif
#endif  // CCE_RUNTIME_RT_MODEL_H
