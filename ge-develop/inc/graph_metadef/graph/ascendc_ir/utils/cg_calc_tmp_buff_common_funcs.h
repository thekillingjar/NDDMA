/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_CG_CALC_TMP_BUFF_COMMON_FUNCS_H
#define METADEF_CXX_CG_CALC_TMP_BUFF_COMMON_FUNCS_H

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/symbolizer/symbolic.h"

inline std::vector<std::unique_ptr<ge::TmpBufDesc>> SameTmpBufSizeWithFirstInput(const ge::AscNode &node) {
  std::vector<std::unique_ptr<ge::TmpBufDesc>> tmp_buf_descs;
  ge::AscNodeInputs node_inputs = node.inputs;
  if (node_inputs.Size() <= 0) {
    return tmp_buf_descs;
  }
  auto expr = ge::Expression(ge::Symbol(ge::GetSizeByDataType(node_inputs[0].attr.dtype)));
  for (const auto &repeat : node_inputs[0].attr.repeats) {
    expr = ge::sym::Mul(expr, repeat);
  }
  tmp_buf_descs.emplace_back(std::make_unique<ge::TmpBufDesc>(ge::TmpBufDesc{expr}));
  return tmp_buf_descs;
}


#endif // METADEF_CXX_CG_CALC_TMP_BUFF_COMMON_FUNCS_H
