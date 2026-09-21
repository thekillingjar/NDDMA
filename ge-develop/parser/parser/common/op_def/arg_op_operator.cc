/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/common/op_def/arg_op_operator.h"
#include <string>
#include "framework/common/fmk_types.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ArgOpOperator::ArgOpOperator() : ParserOperator("Data") {}

ArgOpOperator::~ArgOpOperator() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ArgOpOperator &ArgOpOperator::Index(int64_t index) {
  Attr("index", static_cast<int64_t>(index));

  return *this;
}

int64_t ArgOpOperator::GetIndex() const { return GetIntAttr("index"); }
}  // namespace ge
