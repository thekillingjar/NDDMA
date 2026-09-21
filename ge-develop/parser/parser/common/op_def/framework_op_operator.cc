/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op_def/framework_op_operator.h"
#include <string>
#include "framework/common/fmk_types.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator::FrameworkOpOperator()
    : ParserOperator("FrameworkOp") {}

FrameworkOpOperator::~FrameworkOpOperator() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::Index(int64_t index) {
  Attr(RETVAL_ATTR_NAME_INDEX, static_cast<int64_t>(index));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::NodeDefPkg(
    const std::string &nodedef_pkg) {
  Attr_bt(ATTR_NAME_FRAMEWORK_NODE_DEF, nodedef_pkg);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::Frameworktype(
    int64_t framework_type) {
  Attr(ATTR_NAME_FRAMEWORK_FWK_TYPE, static_cast<int64_t>(framework_type));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::TfOpDef(
    const std::string &opdef_string) {
  Attr(ATTR_NAME_FRAMEWORK_OP_DEF, opdef_string);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::OriginalType(
    const std::string &type) {
  Attr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FrameworkOpOperator &FrameworkOpOperator::FuncDefPkg(const std::string &func_string) {
  Attr_bt(ATTR_NAME_FRAMEWORK_FUNC_DEF, func_string);
  return *this;
}

FMK_FUNC_HOST_VISIBILITY int64_t FrameworkOpOperator::GetFrameworkType() const {
  return GetIntAttr(ATTR_NAME_FRAMEWORK_FWK_TYPE);
}

FMK_FUNC_HOST_VISIBILITY std::string FrameworkOpOperator::GetNodeDefPkg() const {
  return GetStringAttr(ATTR_NAME_FRAMEWORK_NODE_DEF);
}
}  // namespace ge
