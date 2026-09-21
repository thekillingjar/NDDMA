/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/types.h"
#include "ge_graph_dsl/ge.h"

GE_NS_BEGIN

REGISTER_OPTYPE_DEFINE(DATA, "Data");
REGISTER_OPTYPE_DEFINE(HCOMALLGATHER, "HcomAllGather");
REGISTER_OPTYPE_DEFINE(VARIABLE, "Variable");
REGISTER_OPTYPE_DEFINE(CONSTANT, "Const");
REGISTER_OPTYPE_DEFINE(CONSTANTOP, "Constant");
REGISTER_OPTYPE_DEFINE(LESS, "Less");
REGISTER_OPTYPE_DEFINE(MUL, "Mul");
REGISTER_OPTYPE_DEFINE(NETOUTPUT, "NetOutput");
REGISTER_OPTYPE_DEFINE(ADD, "Add");
REGISTER_OPTYPE_DEFINE(WHILE, "While");
REGISTER_OPTYPE_DEFINE(ENTER, "Enter");
REGISTER_OPTYPE_DEFINE(MERGE, "Merge");
REGISTER_OPTYPE_DEFINE(LOOPCOND, "LoopCond");
REGISTER_OPTYPE_DEFINE(SWITCH, "Switch");
REGISTER_OPTYPE_DEFINE(EXIT, "Exit");
REGISTER_OPTYPE_DEFINE(NEXTITERATION, "NextIteration");
REGISTER_OPTYPE_DEFINE(GETNEXT, "GetNext");
REGISTER_OPTYPE_DEFINE(STREAMSWITCH, "StreamSwitch");
REGISTER_OPTYPE_DEFINE(MEMCPYADDRASYNC, "MemcpyAddrAsync");
REGISTER_OPTYPE_DEFINE(LABELSWITCHBYINDEX, "LabelSwitchByIndex");

GE_NS_END
