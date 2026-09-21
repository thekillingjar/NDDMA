/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "framework/common/debug/ge_log.h"
namespace fe {
const std::string ATTR_NAME_UNKNOWN_SHAPE_OP = "_unknown_shape";
bool UnknownShapeUtils::IsUnKnownShapeTensor(const ge::OpDesc &op_desc) {
  for (auto &tenosr_desc_ptr : op_desc.GetAllInputsDescPtr()) {
    if (tenosr_desc_ptr == nullptr) {
      continue;
    }
    if (tenosr_desc_ptr->GetShape().IsUnknownShape()) {
      return true;
    }
  }

  for (auto &tenosr_desc_ptr : op_desc.GetAllOutputsDescPtr()) {
    if (tenosr_desc_ptr == nullptr) {
      continue;
    }
    if (tenosr_desc_ptr->GetShape().IsUnknownShape()) {
      return true;
    }
  }

  return false;
}

bool UnknownShapeUtils::IsUnknownShapeOp(const ge::OpDesc &op_desc) {
  bool unknown_shape_status = false;
  if (ge::AttrUtils::GetBool(op_desc, ATTR_NAME_UNKNOWN_SHAPE_OP, unknown_shape_status)) {
    return unknown_shape_status;
  }
  if (op_desc.GetAllInputsSize() != 0 || op_desc.GetOutputsSize() != 0) {
    unknown_shape_status = IsUnKnownShapeTensor(op_desc);
  }
  ge::OpDesc *no_const_op_desc = const_cast<ge::OpDesc *>(&op_desc);
  (void)ge::AttrUtils::SetBool(*no_const_op_desc, ATTR_NAME_UNKNOWN_SHAPE_OP, unknown_shape_status);
  GELOGD("Op[%s, %s] Set attr unknown_shape [%d].", op_desc.GetName().c_str(), op_desc.GetType().c_str(),
         unknown_shape_status);
  return unknown_shape_status;
}

bool UnknownShapeUtils::IsContainUnknownDimNum(const ge::OpDesc &op_desc) {
  for (auto &ptr : op_desc.GetAllInputsDescPtr()) {
    if (ptr->GetShape().IsUnknownDimNum()) {
      GELOGD("Op[name:%s,type:%s] has an input tensor whose shape contains -2.", op_desc.GetName().c_str(),
             op_desc.GetType().c_str());
      return true;
    }
  }

  for (auto &ptr : op_desc.GetAllOutputsDescPtr()) {
    if (ptr->GetShape().IsUnknownDimNum()) {
      GELOGD("Op[name=%s, type=%s] has an output tensor whose shape contains -2.", op_desc.GetName().c_str(),
             op_desc.GetType().c_str());
      return true;
    }
  }

  return false;
}

bool UnknownShapeUtils::IsUnknownShapeValue(const int64_t &value) {
  if (value == ge::UNKNOWN_DIM || value == ge::UNKNOWN_DIM_NUM) {
    return true;
  }
  return false;
}
}  // namespace fe
