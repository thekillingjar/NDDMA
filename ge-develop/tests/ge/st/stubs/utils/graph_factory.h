/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_22AB7AD26BB44415A28DDF92D6FB08D5
#define INC_22AB7AD26BB44415A28DDF92D6FB08D5

#include "ge_running_env/fake_ns.h"
#include "graph/graph.h"
FAKE_NS_BEGIN

struct GraphFactory{
  static Graph SingeOpGraph();
  static Graph SingeOpGraph2();
  static Graph VariableAddGraph();
  static Graph SingeOpGraph3();
  static Graph BuildRefreshWeight();
  static Graph GraphDataToNetoutput();
  static Graph SingeOpGraphWithSkipAttr();
  static Graph HybridSingeOpGraph();
  static Graph HybridSingeOpGraph2();
  static Graph BuildAicpuSingeOpGraph();
  static Graph BuildVarInitGraph1();
  static Graph BuildRefDataInitGraph1();
  static Graph BuildCheckpointGraph1();
  static Graph BuildVarTrainGraph1();
  static Graph BuildRefDataTrainGraph1();
  static Graph BuildRefDataWithStroageFormatTrainGraph1(Format storage_format, Format origin_format,
                                                        const std::vector<int64_t> &storage_shape,
                                                        const std::string &expand_dims_rule);
  static Graph BuildVarWriteNoOutputRefGraph1();
  static Graph BuildV1LoopGraph1();
  static Graph BuildV1LoopGraph2_CtrlEnterIn();
  static Graph BuildV1LoopGraph3_CtrlEnterIn2();
  static Graph BuildV1LoopGraph4_DataEnterInByPassMerge();
  static Graph BuildGraphForMergeShapeNPass();
  static Graph HybridSingeOpGraphForHostMemInput();
  static Graph HybridSingeOpGraphAicpuForHostMemInput();
  static Graph BuildInputMergeCopyGraph();
  static Graph BuildStaticInputGraph();
  static Graph BuildDynamicInputGraph();
  static Graph BuildDynamicInputGraphWithVarAndConst();
  static Graph BuildDynamicInputGraphWithVarAndConst2();
  static Graph BuildEnhancedUniqueGraph();
  static Graph BuildDynamicAndBoarderInputGraph();
  static Graph BuildOutPutGraph();
};

FAKE_NS_END

#endif
