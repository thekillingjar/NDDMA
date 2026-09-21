/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mode_b_index_generator.h"
#include <cstdint>
#include <iostream>
#include <securec.h>
#include "log.h"

namespace amctcmp {
ModeBIndexGenerator::ModeBIndexGenerator(char* dataBuffer, char* indexBuffer, bool isTight, size_t fractalSize,
    size_t initOffset) : IndexGenerator(dataBuffer, indexBuffer, isTight, fractalSize, initOffset)
{
    for (int i = 0; i < modebRamChannel_; i++) {
        balanceFact_[i] = 0;
    }
}
CmpStatus ModeBIndexGenerator::GenerateIndexAndInsertCmpData(const char* cmpData, size_t cmpLen,  CompressType type)
{
    int result;
    CompressInfoValT cmpInfoTmp;

    size_t cmpAlignCount = (cmpLen + compressLenAlign_ - 1) / compressLenAlign_;
    cmpInfoTmp.dataLen = static_cast<uint16_t>(cmpAlignCount & compressTenBit_);
    cmpInfoTmp.extended = static_cast<uint16_t>((cmpAlignCount >> compressInfoDataLenBits_) & compressOneBit_);
    cmpInfoTmp.modeFlag = 0;
    cmpInfoTmp.circleMode = 0;
    cmpInfoTmp.storeOffset = 0;
    cmpInfoTmp.specialFlag = 1;

    cmpInfoTmp.reserved = 0;
    cmpInfoTmp.offset = 0;

    if (type == BYPASS) {
        Log("type is bypass");
        cmpInfoTmp.specialFlag = 0;
    }
    if (isTight_) {
        Log("Offset is: " << cursor_ + initOffset_ << ", " << "Alignment is " << compressLenAlign_);
        cmpInfoTmp.offset = ((cursor_ + initOffset_ + compressLenAlign_ - 1) / compressLenAlign_);
        result = memcpy_s(dataBuffer_ + cursor_, cmpLen, cmpData, cmpLen);
        if (result != EOK) {
            LogFatal("Memcpy data failed");
            return RET_ERROR;
        }
        cursor_ += ((cmpLen + compressLenAlign_ - 1) / compressLenAlign_) * compressLenAlign_;

        size_t offset = sizeof(cmpInfoTmp)*iterNum_;
        result = memcpy_s(indexBuffer_ + offset, sizeof(cmpInfoTmp), &cmpInfoTmp, sizeof(cmpInfoTmp));
        if (result != EOK) {
            LogFatal("Memcpy data failed");
            return RET_ERROR;
        }
    } else  {
        int span = (cmpLen + offsetSize_ - 1) / offsetSize_;
        int totalSpan = fractalSize_ / offsetSize_;
        int diff = totalSpan - span;
        int minValue = balanceFact_[0];
        int minIndex = 0;

        for (int i = 0; i <= diff && i < modebRamChannel_; i++) {
            if (balanceFact_[i] < minValue) {
                minIndex = i;
                minValue = balanceFact_[i];
            }
        }

        for (int i = 0; i < span; i++) {
            balanceFact_[(minIndex + i) % modebRamChannel_]++;
        }

        cmpInfoTmp.storeOffset = minIndex;

        result = memcpy_s(dataBuffer_ + cursor_ + cmpInfoTmp.storeOffset * offsetSize_, cmpLen, cmpData, cmpLen);
        if (result != EOK) {
            LogFatal("Memcpy data failed");
            return RET_ERROR;
        }

        cursor_ += fractalSize_;
        result = memcpy_s(indexBuffer_ + iterNum_ * notTightIndexLen_, notTightIndexLen_,
            &cmpInfoTmp, notTightIndexLen_);
        if (result != EOK) {
            LogFatal("Memcpy data failed");
            return RET_ERROR;
        }
    }

    iterNum_++;
    compressedLength_ += ((cmpLen + compressLenAlign_ - 1) / compressLenAlign_) * compressLenAlign_;
    return RET_SUCCESS;
}
} // amctcmp
