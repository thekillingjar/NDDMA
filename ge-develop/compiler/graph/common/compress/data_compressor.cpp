/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_compressor.h"
#include <vector>
#include <climits>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <memory>
#include <securec.h>
#include "log.h"

namespace {
bool FreqCmp(std::pair<int, size_t> i, std::pair<int, size_t> j)
{
    return i.second > j.second;
}
}

namespace amctcmp {
using  uchar = unsigned char;

constexpr uint32_t CHAR_SIZE = 256;
constexpr uint32_t HEADER_LEN = 8;
constexpr uint32_t CMP_BLOCK_SIZE = 8;
constexpr uint32_t CMP_BLOCK_SIZE_BITS = CMP_BLOCK_SIZE * CHAR_BIT;
constexpr uint32_t CODE_LEN_BASE = 3;
constexpr uint32_t END_MARK = 503;
constexpr uint32_t END_MARK_LEN = CODE_LEN_BASE + CODE_LEN_BASE + CODE_LEN_BASE;
constexpr uint32_t PREFIX_CODE = 15;
constexpr uint32_t PREFIX_CODE_LEN = 4;
constexpr uint32_t XOR_CHECK_OFFSET = 2;
constexpr uint32_t LOW_SPARSE_HEAD_OVERLAPPING = 4;
constexpr uint32_t HIGH_SPARSE_HEAD_OVERLAPPING = 2;
constexpr uint32_t LOW_SPARSE_HEAD_DICT_CODE = 0xA8;
constexpr uint32_t HIGH_SPARSE_HEAD_DICT_CODE = 0x88;
constexpr uint32_t LOW_SPARSE_ZERO_DICT_LEN = 2;
constexpr uint32_t HIGH_SPARSE_ZERO_DICT_LEN = 5;
constexpr uint32_t RLE_9 = 9;
constexpr uint32_t REVERSE_SHIFT = 16;


inline char BitReverse(char a)
{
    return char(
        ((((uchar(a) * 0x0802LU) & 0x22110LU) | ((uchar(a) * 0x8020LU) & 0x88440LU)) * 0x10101LU) >> REVERSE_SHIFT);
}

// run-length code dict, be reused by low_sparse and high_sparse
const int DataCompressor::relZeroCode_[RLE_ZERO_COUNT] = {0, 4, 2, 6, 5};
const int DataCompressor::relZeroCodeLen_[RLE_ZERO_COUNT] = {3, 3, 3, 3, 6};
const int DataCompressor::relZeroCodeCount_[RLE_ZERO_COUNT] = {1, 2, 3, 4, 9};

const int DataCompressor::lowSparseCode_[LOW_SPARSE_DICT_SIZE] = {
    0, 2, 6, 1, 5, 37, 21, 53, 13, 45, 29, 61, 3, 35, 19, 51, 11, 43, 27, 59,
    7, 39, 23, 279, 151, 407, 87, 343, 215, 471, 55, 311, 183, 439, 119, 375};
const int DataCompressor::lowSparseCodeLen_[LOW_SPARSE_DICT_SIZE] = {
    3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

const int DataCompressor::highSparseCode_[HIGH_SPARSE_DICT_SIZE] = {
    0, 1, 37, 21, 53, 13, 45, 29, 61, 3, 35, 19, 51, 11, 43, 27, 59, 7, 39,
    23, 279, 151, 407, 87, 343, 215, 471, 55, 311, 183, 439, 119, 375, 247};
const int DataCompressor::highSparseCodeLen_[HIGH_SPARSE_DICT_SIZE] = {
    3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

DataCompressor::DataCompressor(enum CompressMode mode, const CompressConfig& compressConfig)
    : fileBlockSize_(compressConfig.maxRatio)
{
    // current only support mode_a and mode_b
    if (mode == MODE_A) {
        engineNum_ = 4; // mode a engine number is 4
        cmpAlign_ = 32; // compress alignment is 32
        baseFileLen_ = 128; // base file alignment is 128
    } else {
        engineNum_ = 1; // mode b engine number is 1
        cmpAlign_ = 32; // compress alignment is 32
        baseFileLen_ = 512; // base file alignment is 512
    }

    // current only support low_sparse and high_sparse
    if (compressConfig.compressType == LOW_SPARSE) {
        dictSize_ = LOW_SPARSE_DICT_SIZE;
        dictOffset_ = LOW_SPARSE_HEAD_OVERLAPPING - HIGH_SPARSE_HEAD_OVERLAPPING;
        headOverlap_ = LOW_SPARSE_HEAD_OVERLAPPING;
        zeroDictSize_ = LOW_SPARSE_ZERO_DICT_LEN;
        nonZeroCodePtr_ = lowSparseCode_;
        nonZeroCodeLenPtr_ = lowSparseCodeLen_;
    } else {
        dictSize_ = HIGH_SPARSE_DICT_SIZE;
        dictOffset_ = 0;
        headOverlap_ = HIGH_SPARSE_HEAD_OVERLAPPING;
        zeroDictSize_ = HIGH_SPARSE_ZERO_DICT_LEN;
        nonZeroCodePtr_ = highSparseCode_;
        nonZeroCodeLenPtr_ = highSparseCodeLen_;
    }
}

void DataCompressor::InitDict(const char* data, size_t len)
{
    std::vector<size_t> freq(CHAR_SIZE, 0);
    for (size_t i = 0; i < len; i++) {
        freq[uchar(data[i])]++;
    }

    std::vector<std::pair<int, size_t>> indexFreq(CHAR_SIZE);
    for (uint32_t i = 0; i < CHAR_SIZE; i++) {
        indexFreq[i].first = i;
        indexFreq[i].second = freq[i];
    }

    sort(indexFreq.begin(), indexFreq.end(), FreqCmp);

    Log("Freqency List:");
    dict_.resize(dictSize_);
    for (uint32_t i = 0; i < dictSize_; i++) {
        int code = indexFreq[i].first;
        dict_[i] = code;
        ranks_.emplace(code);
        codeMap_.emplace(code, i);
        Log("rank:"<<i<<" code: "<<std::hex<<indexFreq[i].first<<" frequency: "<<std::dec<<indexFreq[i].second);
    }

    zeroChar_ = dict_[0];
}

CmpStatus DataCompressor::SplitInput(const char* inputData, size_t inputLen, char* outputData) const
{
    Log("engineNum is " << engineNum_);
    size_t loopNum = inputLen / (engineNum_ * fileBlockSize_);
    size_t outputOffset = 0;
    for (size_t i = 0; i < engineNum_; i++) {
        for (size_t j = 0; j < loopNum; j++) {
            size_t inputOffset = engineNum_ * fileBlockSize_ * j + i * fileBlockSize_;
            auto ret = memcpy_s(outputData + outputOffset, fileBlockSize_, inputData + inputOffset, fileBlockSize_);
            if (ret != EOK) {
                LogFatal("Memcpy data failed");
                return RET_ERROR;
            }
            outputOffset += fileBlockSize_;
        }
    }
    return RET_SUCCESS;
}

bool DataCompressor::InsertCompressCode(uint64_t& code, uint32_t& codeLen, int tempCode, uint32_t tempCodeLen) const
{
    // after insert code's length will exceed 64bits, get into next cpacket compress
    if (codeLen + tempCodeLen > CMP_BLOCK_SIZE_BITS) {
        return false;
    }
    code |= static_cast<uint64_t>(tempCode) << codeLen;
    codeLen += tempCodeLen;
    return true;
}

void DataCompressor::CompressZeroChar(uint32_t& countZero, uint64_t& code, uint32_t& codeLen, bool& hasGap) const
{
    for (int i = zeroDictSize_ - 1; i >= 0; i--) {
        uint32_t quotient = countZero / relZeroCodeCount_[i];
        while (quotient > 0) {
            int tempCode = relZeroCode_[i];
            uint32_t tempCodeLen = relZeroCodeLen_[i];
            hasGap = InsertCompressCode(code, codeLen, tempCode, tempCodeLen);
            // rle_9 -> 6bits, other rle -> 3bits. Insert len should be as close as possible 64bits
            if (!hasGap && relZeroCodeCount_[i] != RLE_9) {
                return;
            } else if (!hasGap && relZeroCodeCount_[i] == RLE_9) {
                break;
            }
            quotient--;
            countZero -= relZeroCodeCount_[i];
        }
    }
    return;
}

void DataCompressor::EncodeNonZero(
    char temp, uint32_t& codeLen, uint64_t& code, uint32_t& countInput, bool& hasGap) const
{
    int tempCode;
    uint32_t tempCodeLen;
    if (ranks_.find(uchar(temp)) != ranks_.end()) {
        // avoid upper level caller's logic error
        if (codeMap_.at(uchar(temp)) <= 0) {
            LogFatal("EncodeNonZero found zeroCode in input data.");
            return;
        }
        // encode according to dict
        tempCode = nonZeroCodePtr_[codeMap_.at(uchar(temp))];
        tempCodeLen = nonZeroCodeLenPtr_[codeMap_.at(uchar(temp))];
    } else {
        // no dict:1111+ASCII
        tempCode = (uchar(BitReverse(temp)) << PREFIX_CODE_LEN) | uchar(PREFIX_CODE);
        tempCodeLen = 4 * CODE_LEN_BASE; // 4 code len for non-dict encoding
    }

    hasGap = InsertCompressCode(code, codeLen, tempCode, tempCodeLen);
    if (hasGap) {
        countInput++;
    }
    return;
}

uint32_t DataCompressor::EncodeBlock(
    const char* inputData, uint32_t inputLen, uint64_t& code, bool& hasGap, uint32_t& countInput) const
{
    uint32_t codeLen = 0;
    uint32_t countZero = 0;
    for (uint32_t inputIndex = 0; inputIndex < inputLen; inputIndex++) {
        char temp = inputData[inputIndex];
        if (zeroChar_ == temp) {
            countZero++;
        }
        // zero char run-length encoding
        if (countZero > 0 && ((zeroChar_ != temp) || (inputIndex == inputLen - 1))) {
            countInput += countZero;
            CompressZeroChar(countZero, code, codeLen, hasGap);
            countInput -= countZero;
            if (!hasGap) {
                break;
            }
        }
        // non zero char encoding
        if (zeroChar_ != temp) {
            EncodeNonZero(temp, codeLen, code, countInput, hasGap);
            if (!hasGap) {
                break;
            }
        }
    }
    return codeLen;
}

int DataCompressor::CompressDataBlock(
    const char* inputData, uint32_t inputLen, uint32_t& originOffset, char* outputData) const
{
    uint64_t code = 0;
    bool hasGap = true;
    uint32_t countInput = 0;
    uint32_t codeLen = EncodeBlock(inputData, inputLen, code, hasGap, countInput);
    if (!hasGap) {
        code |= uint64_t(UINT64_MAX << codeLen);
    }
    for (uint32_t i = 0; i < CMP_BLOCK_SIZE; i++) {
        outputData[i] = uchar(code & 0xff);
        code = code >> CHAR_BIT;
    }
    originOffset += countInput;
    return codeLen;
}

CmpStatus DataCompressor::CompressSingleEngineData(
    const char* inputData, uint32_t inputLen, char* outputData, uint32_t& outputLen, CompressType& type) const
{
    std::unique_ptr<char[]> fetchData(new (std::nothrow) char[fileBlockSize_]);
    if (fetchData == nullptr) {
        LogFatal("Failed to apply for data storage space.");
        return RET_ERROR;
    }

    uint32_t countBits = 0;
    uint32_t inputOffset = 0;
    while (inputOffset < inputLen) {
        if (outputLen >= inputLen) {
            type = BYPASS;
            return RET_SUCCESS;
        }
        uint32_t actualFetchByte = fileBlockSize_ > (inputLen - inputOffset) ? inputLen - inputOffset : fileBlockSize_;
        (void)memset_s(fetchData.get(), fileBlockSize_, 0, fileBlockSize_);
        if (memcpy_s(fetchData.get(), actualFetchByte, inputData + inputOffset, actualFetchByte) != EOK) {
            LogFatal("Memcpy data failed");
            return RET_ERROR;
        }
        countBits = CompressDataBlock(fetchData.get(), actualFetchByte, inputOffset, outputData + outputLen);
        outputLen += CMP_BLOCK_SIZE;
    }

    uint32_t finalCode = END_MARK;
    int finalCodeLen = END_MARK_LEN;
    // if successfully compressed, conceal the package
    if ((outputLen >= inputLen) && (countBits + finalCodeLen > CMP_BLOCK_SIZE_BITS)) {
        type = BYPASS;
    } else if (countBits + finalCodeLen <= CMP_BLOCK_SIZE_BITS) {
        // if the length of last cpaket + len(end_mark) 9 bits < len(cpaket)=64bits,
        // push end_mark into the last cpacket
        // the calculate the real length including end_mark
        uint32_t quotient = countBits / CHAR_BIT;
        uint32_t remain = countBits % CHAR_BIT;
        auto output = reinterpret_cast<uchar *>(outputData + outputLen - (CMP_BLOCK_SIZE - quotient));
        *output |= uchar((finalCode << remain) & 0xff);
        finalCode = finalCode >> (CHAR_BIT - remain);
        *(outputData + outputLen - (CMP_BLOCK_SIZE - quotient) + 1) = finalCode;
        outputLen -= CMP_BLOCK_SIZE - ((countBits + finalCodeLen + CHAR_BIT - 1) / CHAR_BIT);
    } else if (outputLen < inputLen) {
        // first, append 1 in the current cpaket
        *(outputData + outputLen - 1) |= uchar(0xff << (countBits % CHAR_BIT));
        // Add one cpaket and write END_MARK
        // 0~7 bit
        *(outputData + outputLen) = uchar(finalCode & 0xff);
        outputLen++;
        // 8bit
        finalCode = finalCode >> CHAR_BIT;
        *(outputData + outputLen) = uchar(finalCode & 0xff);
        outputLen++;
    }
    return RET_SUCCESS;
}

CmpStatus DataCompressor::CompressBypassMode(const char* inputData, size_t inputLen, char* outputData) const
{
    Log("fall back to bypass");
    size_t len = inputLen * sizeof(char);
    if (memcpy_s(outputData, len, inputData, len) != EOK) {
        LogFatal("Memcpy data failed");
        return RET_ERROR;
    }
    return RET_SUCCESS;
}

void DataCompressor::InsertCompressHeader(
    const char* inputData, uint32_t inputLen, char* outputData, CompressType type) const
{
    if (type == BYPASS) {
        return;
    }
    for (uint32_t i = 0; i < engineNum_; i++) {
        char* cmpPtr = outputData + i * inputLen / engineNum_;
        char* header = cmpPtr;
        const char* inPtr = inputData + i * fileBlockSize_;
        // generate check-code
        uint32_t checksumHigh = 0;
        uint32_t checksumLow = 0;
        for (uint32_t j = 0; j < inputLen / engineNum_; j += XOR_CHECK_OFFSET, inPtr += XOR_CHECK_OFFSET) {
            checksumHigh ^= uchar(*inPtr);
            checksumLow ^= uchar(*(inPtr + 1));
            // skip 1 byte per time: 2
            if ((j + XOR_CHECK_OFFSET) % fileBlockSize_ == 0) {
                inPtr += engineNum_ * fileBlockSize_ - fileBlockSize_;
            }
        }
        // mode mark
        if (type == LOW_SPARSE) {
            *cmpPtr++ = static_cast<char>(LOW_SPARSE_HEAD_DICT_CODE & 0xff);
            *cmpPtr++ = uchar(inputLen / engineNum_ / baseFileLen_);
            *cmpPtr++ = dict_[0];
            *cmpPtr++ = dict_[1];
        } else if (type == HIGH_SPARSE) {
            *cmpPtr++ = static_cast<char>(HIGH_SPARSE_HEAD_DICT_CODE & 0xff);
            *cmpPtr++ = uchar(inputLen / engineNum_ / baseFileLen_);
            *cmpPtr++ = 0x00;
            *cmpPtr++ = 0x00;
        }
        // checksum
        *cmpPtr++ = checksumLow;
        *cmpPtr++ = checksumHigh;
        // compress dict
        for (uint32_t j = dictOffset_; j < dictSize_; j++) {
            *cmpPtr++ = dict_[j];
        }
        Log("Header for " << i << "th engine: ");
        LogHexBuffer(header, HEADER_LEN + dictSize_ - headOverlap_);
    }
    return;
}

CmpStatus DataCompressor::ReorderCompressData(
    char* inputData, uint32_t inputLen, uint32_t* engineLen, size_t& outputLen, CompressType type) const
{
    if (inputLen == 0) {
        LogFatal("Data length is 0");
        return RET_ERROR;
    }
    std::unique_ptr<char[]> reorderData(new (std::nothrow) char[inputLen]);
    if (reorderData == nullptr) {
        LogFatal("Failed to apply for data storage space.");
        return RET_ERROR;
    }
    (void)memset_s(reorderData.get(), inputLen,  0, inputLen);

    uint32_t singleLenMax = *std::max_element(engineLen, engineLen + engineNum_);
    if (type == BYPASS) {
        singleLenMax = inputLen / engineNum_;
    }

    uint32_t blockNum = (singleLenMax + cmpAlign_ - 1) / cmpAlign_;
    Log("block num is :" << blockNum << " cmp align is " << cmpAlign_);

    for (uint32_t i = 0; i < engineNum_; i++) {
        for (uint32_t j = 0; j < blockNum * cmpAlign_; j++) {
            uint32_t index = j / cmpAlign_ * cmpAlign_ * engineNum_ + i * cmpAlign_ + j % cmpAlign_;
            reorderData.get()[index] = inputData[j + i * inputLen / engineNum_];
        }
    }

    size_t len = blockNum * cmpAlign_ * engineNum_;
    if (memcpy_s(inputData, len, reorderData.get(), len) != EOK) {
        LogFatal("Memcpy data failed");
        return RET_ERROR;
    }
    outputLen = len;
    return RET_SUCCESS;
}

CmpStatus DataCompressor::CompressFile(
    const char* inputData, size_t inputLen, char* outputData, size_t& outputLen, CompressType& type) const
{
    uint32_t len = inputLen * sizeof(char);
    if (len == 0) {
        LogFatal("Data length is 0");
        return RET_ERROR;
    }

    std::unique_ptr<char[]> splitData(new (std::nothrow) char[len]);
    std::unique_ptr<char[]> cmpData(new (std::nothrow) char[len]);
    if (splitData == nullptr || cmpData == nullptr) {
        LogFatal("Failed to apply for data storage space.");
        return RET_ERROR;
    }

    SplitInput(inputData, inputLen, splitData.get());
    (void)memset_s(cmpData.get(), len, 0, len);

    // always has a dict in header
    uint32_t compressOffset = HEADER_LEN + dictSize_ - headOverlap_;
    Log("Header offset is: " << compressOffset);
    std::vector<uint32_t> singleEngineCompressLen(engineNum_, compressOffset);

    for (uint32_t i = 0; i < engineNum_; i++) {
        uint32_t singleEngineLen = inputLen / engineNum_;
        const char* inputStart = splitData.get() + singleEngineLen * i;
        char* outputStart = cmpData.get() + singleEngineLen * i;
        auto ret = CompressSingleEngineData(inputStart, singleEngineLen, outputStart, singleEngineCompressLen[i], type);
        if (ret != RET_SUCCESS) {
            LogFatal("CompressSingleEngineData failed");
            return ret;
        }

        // if compression ratio > 1, fall back to bypass mode
        if (type == BYPASS) {
            if (CompressBypassMode(splitData.get(), inputLen, cmpData.get()) == RET_SUCCESS) {
                break;
            }
            LogFatal("CompressBypassMode failed");
            return RET_ERROR;
        }
    }

    InsertCompressHeader(inputData, inputLen, cmpData.get(), type);

    size_t tmpBufferLen = outputLen;
    if (ReorderCompressData(cmpData.get(), inputLen, singleEngineCompressLen.data(), outputLen, type) != RET_SUCCESS) {
        LogFatal("ReorderCompressData failed");
        return RET_ERROR;
    }

    if (tmpBufferLen < outputLen) {
        LogFatal("Compress data length is out of buffer size, compress failed.");
        return RET_ERROR;
    }

    if (memcpy_s(outputData, outputLen, cmpData.get(), outputLen) != EOK) {
        LogFatal("Memcpy data failed");
        return RET_ERROR;
    }

    return RET_SUCCESS;
}

} // amctcmp
