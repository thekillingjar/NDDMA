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
#ifndef DOMI_OP_SHAPE_N_OP_H_
#define DOMI_OP_SHAPE_N_OP_H_
#include "parser/common/op_def/operator.h"
#include "framework/omg/parser/parser_types.h"

namespace ge {
class ShapeNOperator : public ParserOperator {
 public:
  ShapeNOperator();
  ~ShapeNOperator() override;

  ShapeNOperator &N(int64_t n);
  int64_t GetN() const;
  ShapeNOperator &InType(ge::DataType t);
  ge::DataType GetInType() const;
  ShapeNOperator &OutType(ge::DataType t);
  ge::DataType GetOutType() const;
};
}  // namespace ge

#endif  // DOMI_OP_SHAPE_N_OP_H_  AUTO GEN PLEASE DO NOT MODIFY IT
