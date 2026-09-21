/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/check_register.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"
#include "graph/utils/op_type_utils.h"
#include "common/checker.h"

namespace ge {
Status UserInputAndContinuousInputOutputChecker(CheckFuncContext &context) {
  // refdata-assign-PhonyConcat这种结构，在assign后面插入，保证refdata能被assign更新
  bool done = false;
  GE_ASSERT_SUCCESS(MemLayoutConflictUtil::AssignVarInsertIdentity(context,
                                                                   ANCHOR_ATTR_USER_MEMORY_INPUT, done));
  if (done) {
    return SUCCESS;
  }
  // 默认情况是在data后面插入identity
  auto user_in_node_index_io = context.node_a;
  if (OpTypeUtils::IsDataNode(context.node_b.node_->GetType())) {
    user_in_node_index_io = context.node_b;
  }

  context.result.insert(user_in_node_index_io.node_->GetOutAnchor(user_in_node_index_io.index_));
  GE_MEM_LAYOUT_CONFLICT_LOGI(context, user_in_node_index_io);
  return SUCCESS;
}

Status UserInputAndNoPaddingContinuousInputChecker(CheckFuncContext &context) {
  auto pc = context.node_a;
  if (MemLayoutConflictUtil::IsContainTargetType(context.type_b, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT)) {
    pc = context.node_b;
  }
  /*
   *  PhonyConcat 的所有输入是同一个anchor，不需要插入identity
   *              data
   *               /\
   *  assign_slice0 assign_slice1 (inplace)
   *             \   /
   *          PhonyConcat
   */
  if (MemLayoutConflictUtil::AllRealInputsAreTheSameOutAnchor(pc.node_)) {
    return SUCCESS;
  }
  return UserInputAndContinuousInputOutputChecker(context);
}

/**
 * 规避方案： load model withq 场景下，用户输入作为nopadding连续输出的情况，不支持刷新第二个及以后的输出的地址
 * 方案详述： 这种场景下，需要在用户输入和需要nopadding连续输入的算子之间插入 indentity 算子
 * 方案约束：
 *                                                                     data
 *     data                                                              |
 *      |                                                             identity
 *    split  (split_dim=0, no_task, no_padding_continuous_output) =>     |
 *   /  |  \                                                           split
 * op1 op2 op3                                                         / |  \
 *                                                                 op1  op2 op3
 *  split no-task 优化之后， op2、op3 的地址在加载的时候，无法正确的设置地址，需要在 data 和 split 之间插入 identity
**/
Status UserInputAndNoPaddingContinuousOutputChecker(CheckFuncContext &context) {
  if (context.graph_info.is_support_user_input_nopadding_continuous_output) {
    GELOGI("[MemConflict] user input and nopadding continous output is supported. "
           "[%s][%s] and [%s][%s] do not need to insert identity.",
           context.node_a.node_->GetNamePtr(), CheckerLog::ToStr(context.type_a).c_str(),
           context.node_b.node_->GetNamePtr(), CheckerLog::ToStr(context.type_b).c_str());
    return SUCCESS;
  }
  return UserInputAndContinuousInputOutputChecker(context);
}

REGISTER_FUNC(ANCHOR_ATTR_USER_MEMORY_INPUT, ANCHOR_ATTR_CONTINUOUS_OUTPUT,
              UserInputAndContinuousInputOutputChecker);
REGISTER_FUNC(ANCHOR_ATTR_USER_MEMORY_INPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT,
              UserInputAndNoPaddingContinuousOutputChecker);
REGISTER_FUNC(ANCHOR_ATTR_USER_MEMORY_INPUT, ANCHOR_ATTR_CONTINUOUS_INPUT,
              UserInputAndContinuousInputOutputChecker);
REGISTER_FUNC(ANCHOR_ATTR_USER_MEMORY_INPUT, ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT,
              UserInputAndNoPaddingContinuousInputChecker);
}  // namespace ge
