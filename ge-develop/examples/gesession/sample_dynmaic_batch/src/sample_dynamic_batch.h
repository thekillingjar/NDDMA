/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_SAMPLE_DYNAMIC_BATCH_H_
#define CANN_GRAPH_ENGINE_SAMPLE_DYNAMIC_BATCH_H_

#include "ge/ge_api_v2.h"
#include <string>
#include <vector>

using namespace std;

#define INFO_LOG(fmt, ...)  fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...)  fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

enum Result {
    SUCCESS = 0,
    FAILED = 1
};

// read bin file to the allocated memory
Result ReadBinFile(const string &fileName, void *inputData, uint32_t &picDataSize);

Result LoadDataFromFile(const vector<string> &binFiles, const std::initializer_list<int64_t> &dims,
                        vector<gert::Tensor> &inputs);


class SampleDynamicBatch {
public:
    explicit SampleDynamicBatch(const map<ge::AscendString, ge::AscendString> &options);
    ~SampleDynamicBatch();

    Result ParseModelAndBuildGraph(const string &modelPath);

    Result CompileGraph(const map<ge::AscendString, ge::AscendString> &options, const vector<ge::Tensor> &inputs);

    Result Process(const vector<string> &binFiles, const std::initializer_list<int64_t> &dims);

    void OutputModelResult();

private:
    int32_t deviceId_{0};
    uint32_t graphId_{0};
    ge::GeSession session_{{}};
    ge::Graph graph_{"resnet50_dynamic_batch"};
    std::vector<gert::Tensor> inputs_;
    std::vector<gert::Tensor> outputs_;
};

#endif // CANN_GRAPH_ENGINE_SAMPLE_DYNAMIC_BATCH_H_
