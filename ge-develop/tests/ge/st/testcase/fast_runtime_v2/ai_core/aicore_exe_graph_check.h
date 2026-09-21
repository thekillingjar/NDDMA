/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGT_SLICE_TYPE_H_AICORE_EXE_GRAPH_CHECK_H
#define SGT_SLICE_TYPE_H_AICORE_EXE_GRAPH_CHECK_H
#include "runtime/model_v2_executor.h"
#include "engine/aicore/fe_rt2_common.h"
using namespace std;
using namespace ge;
namespace gert {
struct JudgeInfo {
  std::string high_pri_name;
  std::string high_pri_type;
  std::string low_pri_name;
  std::string low_pri_type;
};

struct TestRecord {
  FastNode *node;
  ExecuteGraph *graph;
};

bool Judge2KernelPriority(ExecuteGraphPtr exe_graph, JudgeInfo &judge_info);
}


#endif  // SGT_SLICE_TYPE_H_AICORE_EXE_GRAPH_CHECK_H
