/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MODE_A_INDEX_GENERATOR_H
#define MODE_A_INDEX_GENERATOR_H

#include <cstdint>
#include "index_generator.h"

namespace amctcmp {
class ModeAIndexGenerator : public IndexGenerator {
public:
    static const uint16_t ramChannel_ = 2;
    static const uint16_t offsetSize_ = 256;
    static const uint16_t compressInfoDataLenBits_ = 9;
    static const uint16_t compressInfoAlignFlagBits_ = 2;
    static const uint16_t compressInfoModeFlagBits_ = 1;
    static const uint16_t compressInfoSizeBits_ = 2;
    static const uint16_t compressInfoOffsetFlagBits_ = 1;
    static const uint16_t compressInfoSpecialFlagBits_ = 1;
    static const uint16_t compressLenAlign_ = 128;
    static const uint16_t notTightIndexLen_ = 2;

    struct CompressInfoValT {
        uint16_t dataLen       : compressInfoDataLenBits_;
        uint16_t alignFlag     : compressInfoAlignFlagBits_;
        uint16_t modeFlag      : compressInfoModeFlagBits_;
        uint16_t size          : compressInfoSizeBits_;
        uint16_t offsetFlag    : compressInfoOffsetFlagBits_;
        uint16_t specialFlag   : compressInfoSpecialFlagBits_;
        uint16_t reserved;
        uint32_t offset;
    };
public:
    ModeAIndexGenerator(char* dataBuffer, char* indexBuffer, bool isTight, size_t fractalSize, size_t initOffset);
    CmpStatus GenerateIndexAndInsertCmpData(const char* cmpData, size_t cmpLen,  CompressType type);
    ~ModeAIndexGenerator() override = default;
private:
    int balanceFact_[ramChannel_];
};
} // amctcmp
#endif // MODE_A_INDEX_GENERATOR_H
