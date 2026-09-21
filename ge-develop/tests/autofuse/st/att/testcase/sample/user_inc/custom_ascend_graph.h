/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAMPLEPROJECT_ATT_SAMPLE_USER_INC_CUSTOM_ASCEND_GRAPH_H_
#define SAMPLEPROJECT_ATT_SAMPLE_USER_INC_CUSTOM_ASCEND_GRAPH_H_
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
// only define one graph, if you want to support multi graphs, need to make same adaptor
extern "C" {
void BuildOriginGraph(ge::AscGraph &graph);
void AddScheInfoToGraph(ge::AscGraph &graph);
void AddBuffInfoToGraph(ge::AscGraph &graph);
ge::Status GenerateAscGraphs(std::vector<ge::AscGraph> &graphs);
void GeneratorAttOptions(std::map<std::string, std::string> &options);
}
#endif  // SAMPLEPROJECT_ATT_SAMPLE_USER_INC_CUSTOM_ASCEND_GRAPH_H_
