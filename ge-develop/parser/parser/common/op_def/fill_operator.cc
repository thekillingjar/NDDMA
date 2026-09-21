/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op_def/fill_operator.h"
#include "framework/common/fmk_types.h"

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FillOperator::FillOperator() : ParserOperator("Fill") {}

FMK_FUNC_DEV_VISIBILITY FillOperator::~FillOperator() {}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FillOperator &FillOperator::DataType(int64_t dataType) {
  Attr("T", static_cast<int64_t>(dataType));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FillOperator &FillOperator::Alpha(float alpha) {
  Attr("alpha", static_cast<float>(alpha));
  return *this;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FillOperator &FillOperator::Beta(float beta) {
  Attr("beta", static_cast<float>(beta));
  return *this;
}

int64_t FillOperator::GetDataType() const { return GetIntAttr("T"); }

float FillOperator::GetAlpha() const { return GetFloatAttr("alpha"); }

float FillOperator::GetBeta() const { return GetFloatAttr("beta"); }
}  // namespace ge
