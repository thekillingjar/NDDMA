/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACL_RT_MEMCPY_KIND_H
#define ACL_RT_MEMCPY_KIND_H
/**
 * @ingroup dvrt_mem
 * @brief memory copy type
 */
typedef enum tagRtMemcpyKind {
  RT_MEMCPY_HOST_TO_HOST = 0,  // host to host
  RT_MEMCPY_HOST_TO_DEVICE,    // host to device
  RT_MEMCPY_DEVICE_TO_HOST,    // device to host
  RT_MEMCPY_DEVICE_TO_DEVICE,  // device to device, 1P && P2P
  RT_MEMCPY_MANAGED,           // managed memory
  RT_MEMCPY_ADDR_DEVICE_TO_DEVICE,
  RT_MEMCPY_HOST_TO_DEVICE_EX,  // host  to device ex (only used for 8 bytes)
  RT_MEMCPY_DEVICE_TO_HOST_EX,  // device to host ex
  RT_MEMCPY_DEFAULT,            // auto infer copy dir
  RT_MEMCPY_RESERVED,
} rtMemcpyKind_t;

#endif  // ACL_RT_MEMCPY_KIND_H