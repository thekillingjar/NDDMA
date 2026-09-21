/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mode_a_index_generator.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <securec.h>
#include "compress_type.h"
#include "log.h"

namespace amctcmp {
ModeAIndexGenerator::ModeAIndexGenerator(char* dataBuffer, char* indexBuffer, bool isTight, size_t fractalSize,
    size_t initOffset) : IndexGenerator(dataBuffer, indexBuffer, isTight, fractalSize, initOffset)
{
    for (int i = 0; i < ramChannel_; i++) {
        balanceFact_[i] = 0;
    }
}

// mini
CmpStatus ModeAIndexGenerator::GenerateIndexAndInsertCmpData(const char* cmpData, size_t cmpLen,  CompressType type)
{
    int result;
    CompressInfoValT cmpInfoTmp;

    cmpInfoTmp.alignFlag = 0;
    cmpInfoTmp.modeFlag = 0;
    cmpInfoTmp.size = 0;
    cmpInfoTmp.offsetFlag = 0;
    cmpInfoTmp.specialFlag = 1;
    cmpInfoTmp.reserved = 0;
    cmpInfoTmp.offset = 0;

    cmpInfoTmp.dataLen = static_cast<uint16_t>((cmpLen + compressLenAlign_ - 1) / compressLenAlign_);

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
        result = memcpy_s(indexBuffer_ + sizeof(cmpInfoTmp) * iterNum_, sizeof(cmpInfoTmp),
            &cmpInfoTmp, sizeof(cmpInfoTmp));
        if (result != EOK) {
            LogFatal("Memcpy data failed");
            return RET_ERROR;
        }
    } else  {
        int span = cmpLen / offsetSize_;
        if (span % ramChannel_ != 0 && cmpLen + offsetSize_ <= fractalSize_ && balanceFact_[0] > balanceFact_[1]) {
            cmpInfoTmp.offsetFlag = 1;
            balanceFact_[1]++;
        } else if (span % ramChannel_ != 0) {
            balanceFact_[0]++;
        }
        result = memcpy_s(dataBuffer_ + cursor_ + cmpInfoTmp.offsetFlag * offsetSize_, cmpLen, cmpData, cmpLen);
        if (result != EOK) {
            LogFatal("Memcpy data failed");
            return RET_ERROR;
        }

        cursor_ += fractalSize_;
        result = memcpy_s(indexBuffer_ + notTightIndexLen_ * iterNum_, tightModeLen_, &cmpInfoTmp,
            tightModeLen_);
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
