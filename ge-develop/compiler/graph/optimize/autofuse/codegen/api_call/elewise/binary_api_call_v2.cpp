/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "binary_api_call_v2.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"

namespace codegen {
using namespace std;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;

Status BinaryApiCallV2::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                              const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                              const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                              std::string &result) const {
  auto x1 = inputs[0].get();
  auto x2 = inputs[1].get();
  auto y = outputs[0].get();
  stringstream ss;
  string blk_align;
  GE_CHK_STATUS_RET(KernelUtils::BlkAlign(x1.dtype, blk_align), "Codegen blk align failed");
  ss << this->api_name_ << "("
     << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "
     << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], "
     << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], "
     << blk_align << "("
     << x1.actual_size << "));" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<BinaryApiCallV2> register_unary_api_call("BinaryApiCallV2");
}  // namespace codegen