/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIPP_PARAM_CHECK_H_
#define AIPP_PARAM_CHECK_H_
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <sstream>
#include "common/log_inner.h"
#include "securec.h"
#include "acl/acl_mdl.h"
#include "common/dynamic_aipp.h"
#include "executor/ge_executor.h"
#include "model_desc_internal.h"

namespace acl {
aclError AippScfSizeCheck(const aclmdlAIPP *const aippParmsSet, const size_t batchIndex);
uint64_t GetSrcImageSize(const aclmdlAIPP *const aippParmsSet);
aclError AippParamsCheck(const aclmdlAIPP *const aippParmsSet, const std::string &socVersion);
aclError GetAippOutputHW(const aclmdlAIPP *const aippParmsSet, const size_t batchIndex, const std::string &socVersion,
                         int32_t &aippOutputW, int32_t &aippOutputH);
} // namespace acl
#endif // AIPP_PARAM_CHECK_H_
