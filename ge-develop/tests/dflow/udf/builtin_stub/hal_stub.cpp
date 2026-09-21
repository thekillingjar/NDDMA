/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascend_hal.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

drvError_t halSdmaCopy(DVdeviceptr dst, size_t dst_size, DVdeviceptr src, size_t len) {
    memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<void *>(src), len);
    return DRV_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif
