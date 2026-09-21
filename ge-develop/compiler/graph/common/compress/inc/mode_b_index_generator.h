/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MODE_B_INDEX_GENERATOR_H
#define MODE_B_INDEX_GENERATOR_H

#include <cstdint>
#include "index_generator.h"

namespace amctcmp {
class ModeBIndexGenerator : public IndexGenerator {
public:
    static const uint16_t modebRamChannel_ = 4;
    static const uint16_t offsetSize_ = 256;
    static const uint16_t compressInfoDataLenBits_ = 10;
    static const uint16_t compressInfoExtendedBits_ = 1;
    static const uint16_t compressInfoModeFlagBits_ = 1;
    static const uint16_t compressInfoCircleModeBits_ = 1;
    static const uint16_t compressInfoStoreOffsetBits_ = 2;
    static const uint16_t compressInfoSpecialFlagBits_ = 1;
    static const uint16_t compressLenAlign_ = 32;
    static const uint16_t compressTenBit_ = 0x03FF;
    static const uint16_t compressOneBit_ = 0x0001;
    static const uint16_t notTightIndexLen_ = 2;

    struct CompressInfoValT {
        uint16_t dataLen       : compressInfoDataLenBits_;
        uint16_t extended      : compressInfoExtendedBits_;
        uint16_t modeFlag      : compressInfoModeFlagBits_;
        uint16_t circleMode    : compressInfoCircleModeBits_;
        uint16_t storeOffset   : compressInfoStoreOffsetBits_;
        uint16_t specialFlag   : compressInfoSpecialFlagBits_;
        uint16_t reserved;
        uint32_t offset;
    };

public:
    ModeBIndexGenerator(char* dataBuffer, char* indexBuffer, bool isTight, size_t fractalSize, size_t initOffset);
    CmpStatus GenerateIndexAndInsertCmpData(const char* cmpData, size_t cmpLen,  CompressType type);
    ~ModeBIndexGenerator() override = default;
private:
    int balanceFact_[modebRamChannel_];
};
} // amctcmp
#endif // MODE_B_INDEX_GENERATOR_H
