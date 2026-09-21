/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/local_engine/ops_kernel_store/op/no_op.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "engines/local_engine/ops_kernel_store/op/op_factory.h"

namespace ge {
namespace ge_local {
NoOp::NoOp(const Node &node, RunContext &run_context) : Op(node, run_context) {}

Status NoOp::Run() {
  GELOGD("Node:%s type %s is NoOp, no need generate task.", name_.c_str(), type_.c_str());
  // Do nothing
  return SUCCESS;
}

REGISTER_OP_CREATOR(Data, NoOp);

REGISTER_OP_CREATOR(RefData, NoOp);

REGISTER_OP_CREATOR(QueueData, NoOp);

REGISTER_OP_CREATOR(AippData, NoOp);

REGISTER_OP_CREATOR(NoOp, NoOp);

REGISTER_OP_CREATOR(Variable, NoOp);

REGISTER_OP_CREATOR(Constant, NoOp);

REGISTER_OP_CREATOR(Const, NoOp);

REGISTER_OP_CREATOR(ControlTrigger, NoOp);

REGISTER_OP_CREATOR(Merge, NoOp);

REGISTER_OP_CREATOR(FileConstant, NoOp);

REGISTER_OP_CREATOR(ConstPlaceHolder, NoOp);

// Functional Op.
REGISTER_OP_CREATOR(If, NoOp);
REGISTER_OP_CREATOR(_If, NoOp);
REGISTER_OP_CREATOR(StatelessIf, NoOp);
REGISTER_OP_CREATOR(Case, NoOp);
REGISTER_OP_CREATOR(StatelessCase, NoOp);
REGISTER_OP_CREATOR(While, NoOp);
REGISTER_OP_CREATOR(_While, NoOp);
REGISTER_OP_CREATOR(StatelessWhile, NoOp);
REGISTER_OP_CREATOR(For, NoOp);
REGISTER_OP_CREATOR(PartitionedCall, NoOp);
REGISTER_OP_CREATOR(StatefulPartitionedCall, NoOp);
REGISTER_OP_CREATOR(OpTiling, NoOp);
REGISTER_OP_CREATOR(ConditionCalc, NoOp);
REGISTER_OP_CREATOR(UnfedData, NoOp);
// Data flow ops
REGISTER_OP_CREATOR(Stack, NoOp);
REGISTER_OP_CREATOR(StackPush, NoOp);
REGISTER_OP_CREATOR(StackPop, NoOp);
REGISTER_OP_CREATOR(StackClose, NoOp);

REGISTER_OP_CREATOR(Reshape, NoOp);
REGISTER_OP_CREATOR(Bitcast, NoOp);

REGISTER_OP_CREATOR(PhonyConcat, NoOp);
REGISTER_OP_CREATOR(PhonySplit, NoOp);

REGISTER_OP_CREATOR(Flatten, NoOp);
REGISTER_OP_CREATOR(FlattenV2, NoOp);
REGISTER_OP_CREATOR(ExpandDims, NoOp);
REGISTER_OP_CREATOR(ReFormat, NoOp);
REGISTER_OP_CREATOR(Squeeze, NoOp);
REGISTER_OP_CREATOR(Unsqueeze, NoOp);
REGISTER_OP_CREATOR(SqueezeV2, NoOp);
REGISTER_OP_CREATOR(UnsqueezeV2, NoOp);
REGISTER_OP_CREATOR(SqueezeV3, NoOp);
REGISTER_OP_CREATOR(UnsqueezeV3, NoOp);
}  // namespace ge_local
}  // namespace ge
