/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compress.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>

#include <securec.h>

#include "mode_warehouse.h"
#include "log.h"
#include "data_compressor.h"
#include "index_generator_factory.h"

namespace {
#define MAX_FILE_INPUT_LEN 32*1024
constexpr uint32_t FILE_LEN_RATIO = 512;
constexpr uint32_t RATIO = 3;
constexpr uint32_t TIGHT_LENGTH = 8;
constexpr uint32_t NOT_TIGHT_LENGTH = 2;
constexpr uint32_t MAX_RATIO_8X = 64;
constexpr uint32_t MAX_RATIO_4X = 32;

void SetLogSwitch()
{
    char *logEnv(getenv("CMPLOG"));
    if (logEnv != nullptr) {
        LogSwitch::logSwitch = true;
    } else {
        LogSwitch::logSwitch = false;
    }
}

CmpStatus ParamCheck(const char* input, const char* indexes, const char* cmpData, const CompressConfig& param)
{
    if (input == nullptr || indexes == nullptr || cmpData == nullptr) {
        LogFatal("there is null pointer in parameters");
        return RET_ERROR;
    }

    if ((param.maxRatio != MAX_RATIO_8X) && (param.maxRatio != MAX_RATIO_4X)) {
        LogFatal("maxRatio: "<< param.maxRatio <<" is invalid");
        return RET_ERROR;
    }

    if (param.fractalSize == 0 || param.inputSize % param.fractalSize != 0) {
        LogFatal("fractal size is invalid" << " fractal size is: "
                << param.fractalSize << "input size is: " << param.inputSize);
        return RET_ERROR;
    }

    if ((((param.fractalSize) % FILE_LEN_RATIO) != 0) || (param.fractalSize > MAX_FILE_INPUT_LEN)) {
        LogFatal("inputLen: "<< param.inputSize <<" is invalid");
        return RET_ERROR;
    }

    if ((param.compressType != amctcmp::LOW_SPARSE) && (param.compressType != amctcmp::HIGH_SPARSE)) {
        LogFatal("compressType: "<< param.compressType <<" is invalid");
        return RET_ERROR;
    }
    Log("current compress typeId: " << param.compressType);

    return RET_SUCCESS;
}

amctcmp::CompressMode GetMode(const CompressConfig& param)
{
    // 4 unzip engines and 2-channel ram, typical mode: mini
    if (param.engineNum == 4 && param.channel == 2) {
        return amctcmp::MODE_A;
    } else if (param.engineNum == 1 && param.channel == 4) { // 1 unzip engine and 4 channel, typical mode: lite
        return amctcmp::MODE_B;
    }
    return amctcmp::MODE_INVALID;
}
}

CmpStatus CompressWeights(char* input,
                          const CompressConfig& compressConfig,
                          char* indexs,
                          char* output,
                          size_t& compressedLength)
{
    SetLogSwitch();
    // param check
    if (ParamCheck(input, indexs, output, compressConfig) == RET_ERROR) {
        LogFatal("Parameter-checking failed");
        return RET_ERROR;
    }

    amctcmp::CompressMode mode = GetMode(compressConfig);
    if (mode == amctcmp::MODE_INVALID) {
        LogFatal("Invalid Mode");
        return RET_ERROR;
    }

    amctcmp::DataCompressor fc(mode, compressConfig);
    fc.InitDict(input, compressConfig.inputSize);

    amctcmp::IndexGeneratorFactory indexFactory;
    auto indexGen = indexFactory.GetIndexGenerator(mode, output, indexs, compressConfig);
    if (indexGen == nullptr) {
        LogFatal("Invalid IndexGenerator");
        return RET_ERROR;
    }

    const size_t fractalSize = compressConfig.fractalSize;
    size_t outputTmpLen = fractalSize * RATIO;
    std::unique_ptr<char[]> fractalInput(new (std::nothrow) char[fractalSize]);
    std::unique_ptr<char[]> fractalOutput(new (std::nothrow) char[outputTmpLen]);
    if (fractalInput == nullptr || fractalOutput == nullptr) {
        LogFatal("Failed to apply for fractal storage space.");
        return RET_ERROR;
    }

    size_t iterNum = compressConfig.inputSize / fractalSize;
    for (size_t i = 0; i < iterNum; i++) {
        if (memcpy_s(fractalInput.get(), fractalSize, input + i * fractalSize, fractalSize) != EOK) {
            LogFatal("Memcpy data failed");
            return RET_ERROR;
        }

        // step1: encode data of current fractal size
        size_t fractalOutputLen = outputTmpLen;
        amctcmp::CompressType type = static_cast<amctcmp::CompressType>(compressConfig.compressType);
        auto status = fc.CompressFile(fractalInput.get(), fractalSize, fractalOutput.get(), fractalOutputLen, type);
        if (status == RET_ERROR) {
            LogFatal("Compress data failed");
            return status;
        }

        // step2: generate index and insert both index and compressed data into user buffer
        status = indexGen->GenerateIndexAndInsertCmpData(fractalOutput.get(), fractalOutputLen, type);
        if (status == RET_ERROR) {
            LogFatal("index generate and compression data insert failed");
            return status;
        }
    }

    size_t indexLen = compressConfig.isTight ? TIGHT_LENGTH * iterNum : NOT_TIGHT_LENGTH * iterNum;
    compressedLength = indexGen->GetCompressedLen();

    Log("indexes buffer:");
    LogHexBuffer(indexs, indexLen);
    Log("compressed length is " << compressedLength);

    return RET_SUCCESS;
}