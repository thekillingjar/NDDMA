/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/no_use_reshape_remove_pass.h"

#include <string>
#include <vector>

#include "framework/common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace {
const int32_t kReshapeDataIndex = 0;
const int32_t kReshapeShapeIndex = 1;
}  // namespace
Status NoUseReshapeRemovePass::Run(ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  if (op_desc_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node's op_desc is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] NoUseReshapeRemovePass enter. OpDesc is null.");
    return PARAM_INVALID;
  }
  if (op_desc_ptr->GetType() != RESHAPE) {
    return SUCCESS;
  }
  GELOGI("NoUseReshapeRemovePass enter.");

  bool to_be_deleted = true;
  // compare input and output dims
  if (op_desc_ptr->GetAllInputsSize() == 0U || op_desc_ptr->GetAllOutputsDescSize() == 0U) {
    REPORT_INNER_ERR_MSG("E19999", "Input or Output desc num is zero in node:%s(%s), check invalid",
                       op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Input or output num is zero. node name:%s, input size:%zu, output size:%zu",
           op_desc_ptr->GetName().c_str(), op_desc_ptr->GetAllInputsSize(),
           op_desc_ptr->GetAllOutputsDescSize());
    return INTERNAL_ERROR;
  }
  const auto &input_desc = op_desc_ptr->MutableInputDesc(0);
  const auto &output_desc = op_desc_ptr->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  GE_CHECK_NOTNULL(output_desc);
  std::vector<int64_t> input_4dims = input_desc->GetShape().GetDims();
  std::vector<int64_t> output_4dims = output_desc->GetShape().GetDims();

  if (input_desc->GetShape().IsUnknownShape() || output_desc->GetShape().IsUnknownShape()) {
    GELOGI("Current Reshape %s is unknown shape which should be kept.", op_desc_ptr->GetName().c_str());
    return SUCCESS;
  }

  if (input_4dims.size() != output_4dims.size()) {
    GELOGI("Input and output dim size is not equal.Keep this reshape op.");
    return SUCCESS;
  }

  size_t vec_size = input_4dims.size();
  for (size_t i = 0; i < vec_size; i++) {
    if (input_4dims[i] < 0) {
      GELOGI("Input shape is unknown.Keep this reshape op.");
      return SUCCESS;
    }
    if (input_4dims[i] != output_4dims[i]) {
      to_be_deleted = false;
      break;
    }
  }
  if (to_be_deleted) {
    auto ret = TryRemoveConstShapeInput(node);
    GE_CHK_STATUS_RET_NOLOG(ret);
    GELOGI("NoUseReshapeRemovePass remove useless reshape node:%s", node->GetName().c_str());
    return IsolateAndDeleteNode(node, {kReshapeDataIndex});
  }
  return SUCCESS;
}

Status NoUseReshapeRemovePass::TryRemoveConstShapeInput(ge::NodePtr &reshape_node) {
  auto shape_input_anchor = reshape_node->GetInDataAnchor(kReshapeShapeIndex);
  if (shape_input_anchor == nullptr) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(shape_input_anchor->GetPeerOutAnchor());
  auto shape_input = shape_input_anchor->GetPeerOutAnchor()->GetOwnerNode();
  if (shape_input->GetType() != CONSTANT && shape_input->GetType() != CONSTANTOP) {
    return SUCCESS;
  }
  //   op(x)   const(shape)
  //     \     /
  //     reshape
  GE_CHK_STATUS_RET(DeleteUselessConstAxisNode(shape_input),
                    "Remove const axis input of node:%s failed", reshape_node->GetName().c_str());
  return SUCCESS;
}

REG_PASS_OPTION("NoUseReshapeRemovePass").LEVELS(OoLevel::kO3);
}  // namespace ge
