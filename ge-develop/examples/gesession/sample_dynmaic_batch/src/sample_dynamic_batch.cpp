/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sample_dynamic_batch.h"
#include "ge/ge_api_v2.h"
#include "acl/acl_rt.h"
#include "parser/onnx_parser.h"
#include <vector>
#include <map>
#include <fstream>
#include <numeric>


// read bin file to the allocated memory
Result ReadBinFile(const string &fileName, void *inputData, uint32_t &picDataSize)
{
    std::ifstream binFile(fileName);
    if (!binFile.is_open()) {
        ERROR_LOG("File:%s open failed", fileName.c_str());
        return FAILED;
    }
    binFile.seekg(0, std::ifstream::end);
    picDataSize = binFile.tellg();
    if (picDataSize == 0U) {
        ERROR_LOG("File:%s is empty", fileName.c_str());
        binFile.close();
        return FAILED;
    }
    binFile.seekg(0, std::ifstream::beg);
    binFile.read(static_cast<char *>(inputData), picDataSize);
    binFile.close();
    return SUCCESS;
}

Result LoadDataFromFile(const vector<string> &binFiles, const std::initializer_list<int64_t> &dims,
                        vector<gert::Tensor> &inputs)
{
    const vector<int64_t> inputDims(dims);
    if (binFiles.empty() || inputDims.empty()) {
        ERROR_LOG("Please check input bin file num and input dims");
        return FAILED;
    }

    if (binFiles.size() < inputDims[0]) {
        ERROR_LOG("Input bin file less than batchSize");
        return FAILED;
    }

    int64_t shapeSize = accumulate(inputDims.begin(), inputDims.end(), 1LL, std::multiplies<>());
    const int32_t dataTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
    void *inputData = nullptr;
    int64_t inputDataSize = 0;
    aclError aclRet = aclrtMallocHost(&inputData, shapeSize * dataTypeSize * inputDims[0]);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("Malloc host memory failed.");
        return FAILED;
    }

    for (size_t idx = 0; idx < inputDims[0]; idx++) {
        INFO_LOG("Start to process file:%s", binFiles[idx].c_str());
        uint32_t picDataSize = 0;
        if (ReadBinFile(binFiles[idx], static_cast<uint8_t *>(inputData) + inputDataSize, picDataSize) != SUCCESS) {
            ERROR_LOG("Load file:%s failed", binFiles[idx].c_str());
            aclrtFreeHost(inputData);
            return FAILED;
        }
        inputDataSize += picDataSize;
    }

    const gert::StorageShape tensor_shape(dims, dims);
    const gert::StorageFormat tensor_format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor tensor(tensor_shape, tensor_format, gert::kOnHost, ge::DT_FLOAT, inputData, nullptr);
    inputs.push_back(std::move(tensor));

    return SUCCESS;
}

SampleDynamicBatch::SampleDynamicBatch(const map<ge::AscendString, ge::AscendString> &options)
{
    if (ge::GEInitializeV2(options) != ge::SUCCESS) {
        ERROR_LOG("Initialize ge failed.");
        throw std::runtime_error("Initialize ge failed.");
    }
    INFO_LOG("Initialize ge success");

    if (aclrtSetDevice(deviceId_) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed, device id:%d", deviceId_);
        throw std::runtime_error("Set device failed.");
    }
    INFO_LOG("Set device %d success", deviceId_);
}

SampleDynamicBatch::~SampleDynamicBatch()
{
    for (auto& tensor : inputs_) {
        if (aclrtFreeHost(tensor.GetAddr()) != ACL_SUCCESS) {
            WARN_LOG("Free host memory occur exception");
        }
    }

    if (aclrtResetDevice(deviceId_) != ACL_SUCCESS) {
        WARN_LOG("Reset device occur exception, device id:%d", deviceId_);
    }

    if (ge::GEFinalizeV2() != ge::SUCCESS) {
        WARN_LOG("Finalize ge failed.");
    }
}

Result SampleDynamicBatch::ParseModelAndBuildGraph(const std::string &modelPath)
{
    // aclgrphParseONNX will check the validity of modelPath
    const auto graphRet = ge::aclgrphParseONNX(modelPath.c_str(), {}, graph_);
    return graphRet == ge::SUCCESS ? SUCCESS : FAILED;
}

Result SampleDynamicBatch::CompileGraph(const std::map<ge::AscendString, ge::AscendString> &options,
                                        const std::vector<ge::Tensor> &inputs)
{
    if (session_.AddGraph(graphId_, graph_, options) != ge::SUCCESS) {
        ERROR_LOG("[GeSession] add graph failed, graph id:%d", graphId_);
        return FAILED;
    }
    INFO_LOG("Graph add success");

    if (session_.CompileGraph(graphId_, inputs) != ge::SUCCESS) {
        ERROR_LOG("[GeSession] compile graph failed, graph id:%d", graphId_);
        return FAILED;
    }
    INFO_LOG("Graph compile success");
    return SUCCESS;
}

Result SampleDynamicBatch::Process(const std::vector<std::string> &binFiles,
                                   const std::initializer_list<int64_t> &dims)
{
    if (LoadDataFromFile(binFiles, dims, inputs_) != SUCCESS) {
        ERROR_LOG("LoadDataFromFile failed.");
        return FAILED;
    }

    if (session_.RunGraph(graphId_, inputs_, outputs_) != ge::SUCCESS) {
        ERROR_LOG("[GeSession] run graph failed, graph id:%d", graphId_);
        return FAILED;
    }
    INFO_LOG("Graph run success");
    return SUCCESS;
}

void SampleDynamicBatch::OutputModelResult() {
    for (size_t idx = 0; idx < outputs_.size(); idx++) {
        float *outputData = outputs_[idx].GetData<float>();
        const int64_t outputShapeSize = outputs_[idx].GetShapeSize();
        // resnet50 model output shape is [batchSize, 1000]
        const int64_t dim1 = outputs_[idx].GetStorageShape().GetDim(1U); // the expected dim1 is 1000
        if (dim1 == 0) {
            WARN_LOG("The index:[%zu] output shape is incorrect, please analyse the reason", idx);
            continue;
        }

        for (size_t i = 0; i < outputShapeSize / dim1; i++) {
            INFO_LOG("Result of picture %zu:", i + 1U);
            std::map<float, uint32_t, std::greater<> > resultMap;
            for (uint32_t j = 0; j < dim1; j++) {
                resultMap[*outputData] = j;
                outputData++;
            }

            int cnt = 0;
            for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
                // Print top 5
                if (++cnt > 5) {
                    break;
                }

                INFO_LOG("top %d: index[%u] value[%lf]", cnt, it->second, it->first);
            }
        }
    }

    INFO_LOG("Output data success");
}
