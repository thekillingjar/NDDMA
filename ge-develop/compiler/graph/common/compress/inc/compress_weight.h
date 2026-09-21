/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPRESS_WEIGHT_H
#define COMPRESS_WEIGHT_H

#include <cstdint>
#include "compress.h"

const int SHAPE_SIZE_WEIGHT = 4;

struct CompressOpConfig {
    int64_t wShape[SHAPE_SIZE_WEIGHT];
    size_t compressTilingK;
    size_t compressTilingN;
    struct CompressConfig compressConfig;
};

extern "C" CmpStatus CompressWeightsConv2D(const char *const input,
                                           char *const zipBuffer,
                                           char *const infoBuffer,
                                           CompressOpConfig *const param);
#endif // COMPRESS_WEIGHT_H
