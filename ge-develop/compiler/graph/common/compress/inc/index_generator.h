/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEX_GENERATOR_H
#define INDEX_GENERATOR_H

#include <cuchar>
#include "compress_type.h"
#include "compress.h"

namespace amctcmp {
class IndexGenerator {
public:
    IndexGenerator(char* dataBuffer, char* indexBuffer, bool isTight, size_t fractalSize, size_t initOffset)
        : dataBuffer_(dataBuffer),
          indexBuffer_(indexBuffer),
          isTight_(isTight),
          fractalSize_(fractalSize),
          cursor_(0),
          initOffset_(initOffset),
          compressedLength_(0),
          iterNum_(0) {}
    virtual ~IndexGenerator() {};
    virtual CmpStatus GenerateIndexAndInsertCmpData(const char* cmpData, size_t cmpLen, CompressType type) = 0;
    size_t GetCompressedLen() const
    {
        return compressedLength_;
    }
protected:
    static const size_t tightModeLen_ = 2;
    char* dataBuffer_;
    char* indexBuffer_;
    bool isTight_;
    size_t fractalSize_;
    size_t cursor_;
    size_t initOffset_;
    size_t compressedLength_;
    int iterNum_;
};
} // amctcmp
#endif // INDEX_GENERATOR_H
