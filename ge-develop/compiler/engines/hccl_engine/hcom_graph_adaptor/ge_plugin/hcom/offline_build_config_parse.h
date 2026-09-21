/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OFFLINE_BUILD_CONFIG_PARSE_H
#define OFFLINE_BUILD_CONFIG_PARSE_H

#include "hccl/base.h"
#include "hccl/hcom.h"
#include "graph/utils/node_utils.h"
#include "hcom_log.h"

namespace hccl {
HcclResult GetOffDeviceTypeWithoutDev(DevType &devType);
bool IsOfflineCompilation();
HcclResult GetDeterministic(u8 &deterministic);
}  // namespace hccl

#endif /* OFFLINE_BUILD_CONFIG_PARSE_H */
