/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AUTO GEN PLEASE DO NOT MODIFY IT
#include "common/op_def/var_is_initialized_op_operator.h"
#include <string>
#include <vector>

namespace ge {
VarIsInitializedOpOperator::VarIsInitializedOpOperator() : ParserOperator(ge::parser::VARISINITIALIZEDOP) {}

VarIsInitializedOpOperator::~VarIsInitializedOpOperator() {}

VarIsInitializedOpOperator &VarIsInitializedOpOperator::VectorAttr(const std::string &key,
                                                                   std::vector<int64_t> &value) {
  Attr(key, value);
  return *this;
}
}  // namespace ge
