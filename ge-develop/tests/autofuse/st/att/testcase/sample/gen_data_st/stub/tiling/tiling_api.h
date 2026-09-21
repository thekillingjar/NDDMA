/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAMPLEPROJECT_ATT_SAMPLE_ST_PROJECT_STUB_TILING_TILING_API_H_
#define SAMPLEPROJECT_ATT_SAMPLE_ST_PROJECT_STUB_TILING_TILING_API_H_
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform_ascendc.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TCubeTiling)
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(SoftMaxTiling)
END_TILING_DATA_DEF;
}  // namespace optiling
#endif