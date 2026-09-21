/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/local_engine/ops_kernel_store/op/ge_deleted_op.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "engines/local_engine/ops_kernel_store/op/op_factory.h"
#include "ge_context.h"
#include "ge_local_context.h"
#include "base/err_msg.h"

namespace ge {
namespace ge_local {
GeDeletedOp::GeDeletedOp(const Node &node, RunContext &run_context) : Op(node, run_context) {}

Status GeDeletedOp::Run() {
  static const std::map<std::string, std::string> kOpToExpectedOptimization = {
    {"Size", OO_CONSTANT_FOLDING},
    {"Shape", OO_CONSTANT_FOLDING},
    {"ShapeN", OO_CONSTANT_FOLDING},
    {"Rank", OO_CONSTANT_FOLDING}
    // 可以继续扩展
  };

  // 默认提示信息
  std::string reason = "Node:" + name_ + " type is " + type_ + ", should be deleted by ge. Please check your "
                       "graph optimization settings.";

  auto it = kOpToExpectedOptimization.find(type_);
  if (it != kOpToExpectedOptimization.end()) {
    const std::string &ge_option_key = it->second;
    std::string option_name = ge::GetContext().GetReadableName(ge_option_key);

    // 定制提示信息
    std::string opt_value;
    if (GetThreadLocalContext().GetOption(ge_option_key, opt_value) != GRAPH_SUCCESS || opt_value != "true") {
      reason = "Node:" + name_ + " type is " + type_ +  ", should be deleted ge. But the [" + option_name +  "] "
           "optimization option is disabled, please enable it to remove redundant nodes.";
    } else {
      // Option开启了但节点还存在，提示可能是关联的Pass未生效或通过optimization_switch关闭了
      reason = "Node:" + name_ + " type is " + type_ +  ", should be deleted by ge. But it still exists, "
           "Please check if the [" + option_name + "] optimization option relevant passes are working correctly.";
    }
  }
  REPORT_INNER_ERR_MSG("E19999", "%s", reason.c_str());
  GELOGE(FAILED, "[Delete][Node] %s", reason.c_str());
  // Do nothing
  return FAILED;
}

REGISTER_OP_CREATOR(TemporaryVariable, GeDeletedOp);
REGISTER_OP_CREATOR(DestroyTemporaryVariable, GeDeletedOp);
REGISTER_OP_CREATOR(GuaranteeConst, GeDeletedOp);
REGISTER_OP_CREATOR(PreventGradient, GeDeletedOp);
REGISTER_OP_CREATOR(StopGradient, GeDeletedOp);
REGISTER_OP_CREATOR(Size, GeDeletedOp);
REGISTER_OP_CREATOR(Shape, GeDeletedOp);
REGISTER_OP_CREATOR(ShapeN, GeDeletedOp);
REGISTER_OP_CREATOR(Rank, GeDeletedOp);
REGISTER_OP_CREATOR(_Retval, GeDeletedOp);
REGISTER_OP_CREATOR(ReadVariableOp, GeDeletedOp);
REGISTER_OP_CREATOR(VarHandleOp, GeDeletedOp);
REGISTER_OP_CREATOR(VarIsInitializedOp, GeDeletedOp);
REGISTER_OP_CREATOR(Snapshot, GeDeletedOp);
REGISTER_OP_CREATOR(RefIdentity, GeDeletedOp);
REGISTER_OP_CREATOR(Identity, GeDeletedOp);
REGISTER_OP_CREATOR(IdentityN, GeDeletedOp);
REGISTER_OP_CREATOR(VariableV2, GeDeletedOp);
REGISTER_OP_CREATOR(Empty, GeDeletedOp);
REGISTER_OP_CREATOR(PlaceholderWithDefault, GeDeletedOp);
REGISTER_OP_CREATOR(IsVariableInitialized, GeDeletedOp);
REGISTER_OP_CREATOR(PlaceholderV2, GeDeletedOp);
REGISTER_OP_CREATOR(Placeholder, GeDeletedOp);
REGISTER_OP_CREATOR(End, GeDeletedOp);
REGISTER_OP_CREATOR(Switch, GeDeletedOp);
REGISTER_OP_CREATOR(RefMerge, GeDeletedOp);
REGISTER_OP_CREATOR(RefSwitch, GeDeletedOp);
REGISTER_OP_CREATOR(TransShape, GeDeletedOp);
REGISTER_OP_CREATOR(GatherShapes, GeDeletedOp);
}  // namespace ge_local
}  // namespace ge
