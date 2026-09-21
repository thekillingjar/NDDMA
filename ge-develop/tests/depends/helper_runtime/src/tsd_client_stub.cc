/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu/tsd/tsd_client.h"

uint32_t TsdFileLoad(const uint32_t device_id,
                     const char_t *file_path,
                     const uint64_t path_len,
                     const char_t *file_name,
                     const uint64_t name_len) {
  return 0U;
}

uint32_t TsdFileUnLoad(const uint32_t device_id,
                       const char_t *file_path,
                       const uint64_t path_len) {
  return 0U;
}

uint32_t TsdGetProcListStatus(const uint32_t device_id, ProcStatusParam *status, uint32_t num) {
  return 0U;
}

uint32_t TsdProcessOpen(const uint32_t device_id, ProcOpenArgs *args) {
  if (args != nullptr && args->subPid != nullptr) {
    *args->subPid = 111;
  }
  return 0U;
}

uint32_t ProcessCloseSubProcList(const uint32_t device_id, const ProcStatusParam *status, const uint32_t num) {
  return 0U;
}

uint32_t TsdInitFlowGw(const uint32_t device_id, const InitFlowGwInfo * const info) {
  return 0U;
}

uint32_t TsdCapabilityGet(const uint32_t device_id, const int32_t type, const uint64_t ptr) {
  uint64_t *value = reinterpret_cast<uint64_t *>(static_cast<uintptr_t>(ptr));
  *value = 1;
  return 0U;
}
