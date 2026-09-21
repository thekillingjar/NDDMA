/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FILE_COMPRESSOR_H
#define FILE_COMPRESSOR_H

#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <vector>

#include "compress.h"
#include "mode_warehouse.h"
#include "compress_type.h"

namespace amctcmp {
constexpr uint32_t LOW_SPARSE_DICT_SIZE = 36;
constexpr uint32_t HIGH_SPARSE_DICT_SIZE = 34;
constexpr uint32_t RLE_ZERO_COUNT = 5;

class DataCompressor {
public:
    DataCompressor(enum CompressMode mode,
                   const CompressConfig& compressConfig);
    ~DataCompressor() = default;
    void InitDict(const char* data,
                  size_t len);
    CmpStatus CompressFile(const char* inputData,
                           size_t inputLen,
                           char* outputData,
                           size_t& outputLen,
                           CompressType& type) const;
private:
    CmpStatus SplitInput(const char* inputData,
                         size_t inputLen,
                         char* outputData) const;
    bool InsertCompressCode(uint64_t& code,
                            uint32_t& codeLen,
                            int tempCode,
                            uint32_t tempCodeLen) const;
    void EncodeNonZero(char temp,
                       uint32_t& codeLen,
                       uint64_t& code,
                       uint32_t& countInput,
                       bool& hasGap) const;
    uint32_t EncodeBlock(const char* inputData,
                     uint32_t inputLen,
                     uint64_t& code,
                     bool& hasGap,
                     uint32_t& countInput) const;
    int CompressDataBlock(const char* inputData,
                          uint32_t inputLen,
                          uint32_t& originOffset,
                          char* outputData) const;
    CmpStatus CompressSingleEngineData(const char* inputData,
                                       uint32_t inputLen,
                                       char* outputData,
                                       uint32_t& outputLen,
                                       CompressType& type) const;
    CmpStatus CompressBypassMode(const char* inputData,
                                 size_t inputLen,
                                 char* outputData) const;
    void CompressZeroChar(uint32_t& countZero,
                          uint64_t& code,
                          uint32_t& codeLen,
                          bool& hasGap) const;
    void InsertCompressHeader(const char* inputData,
                                   uint32_t inputLen,
                                   char* outputData,
                                   CompressType type) const;
    CmpStatus ReorderCompressData(char* inputData,
                                  uint32_t inputLen,
                                  uint32_t* engineLen,
                                  size_t& outputLen,
                                  CompressType type) const;
    static const int lowSparseCode_[LOW_SPARSE_DICT_SIZE];
    static const int lowSparseCodeLen_[LOW_SPARSE_DICT_SIZE];
    static const int highSparseCode_[HIGH_SPARSE_DICT_SIZE];
    static const int highSparseCodeLen_[HIGH_SPARSE_DICT_SIZE];
    static const int relZeroCode_[RLE_ZERO_COUNT];
    static const int relZeroCodeLen_[RLE_ZERO_COUNT];
    static const int relZeroCodeCount_[RLE_ZERO_COUNT];
    uint32_t engineNum_;
    uint32_t cmpAlign_;
    uint32_t baseFileLen_;
    size_t fileBlockSize_;
    size_t dictSize_;
    size_t zeroDictSize_;
    size_t dictOffset_;
    size_t headOverlap_;
    const int* nonZeroCodePtr_;
    const int* nonZeroCodeLenPtr_;
    char zeroChar_ = 'a';
    std::vector<char> dict_;
    // code in dict
    std::unordered_set<int> ranks_;
    // code to dict index
    std::unordered_map<int, int> codeMap_;
};
} // amctcmp
#endif // FILE_COMPRESSOR_H
