/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DOMI_OP_FILL_OP_H_
#define DOMI_OP_FILL_OP_H_
#include "parser/common/op_def/operator.h"

namespace ge {
class FillOperator : public ParserOperator {
 public:
  FillOperator();

  ~FillOperator() override;

  FillOperator &DataType(int64_t dataType);

  FillOperator &Alpha(float alpha);

  FillOperator &Beta(float beta);

  int64_t GetDataType() const;

  float GetAlpha() const;

  float GetBeta() const;
};
}  // namespace ge

#endif  // DOMI_OP_FILL_OP_H_
