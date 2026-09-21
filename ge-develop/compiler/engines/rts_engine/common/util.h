/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RTS_ENGINE_COMMON_UTIL_H
#define RTS_ENGINE_COMMON_UTIL_H

#include <cstdint>
#include "runtime/base.h"
#include "external/ge/ge_api_error_codes.h"
#include "graph/utils/node_utils.h"

namespace cce {
namespace runtime {
/**
 * Get soc version by ge context or rtGetSocVersion
 * @return The status whether call interface successfully
 */
ge::Status GetSocVersion(char_t *version, int32_t socVersionLen);

ge::Status IsNeedCalcOpRunningParam(const ge::Node &geNode, bool &isNeed);

ge::Status IsSupportFftsPlus(bool &isSupportFlag);
}  // namespace runtime
}  // namespace cce
#endif

