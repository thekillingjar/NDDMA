/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPRESS_H
#define COMPRESS_H

#include <cuchar>
#include <cstdint>

enum CmpStatus {
    RET_SUCCESS = 0,
    RET_ERROR = -1
};

struct CompressConfig {
    size_t inputSize; // length of data to compress
    size_t engineNum; // how many decompress engines
    size_t maxRatio; // how much size of a basic compression block (8x: 64 4x: 32)
    size_t channel; // channels of L2 or DDR. For load balance
    size_t fractalSize; // size of compressing block
    bool isTight; // whether compose compressed data tightly
    size_t init_offset;
    int32_t compressType = 0; // compress type flag only 0 and 1 supported (0: low_sparse, 1: high_sparse)
};

CmpStatus CompressWeights(char* input,
                          const CompressConfig& compressConfig,
                          char* indexs,
                          char* output,
                          size_t& compressedLength);

CmpStatus SparseWeightsConv2D(const char *const input, size_t weight_size);
#endif  // COMPRESS_H
