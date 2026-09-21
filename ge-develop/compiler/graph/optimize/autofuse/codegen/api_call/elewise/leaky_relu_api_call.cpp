/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "leaky_relu_api_call.h"

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

Status LeakyReluApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();

  std::string x_dtype;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x.dtype, x_dtype), "get name of dtype:%d failed", static_cast<int32_t>(x.dtype));

  stringstream ss;
  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], "
     << "(" << x_dtype << ")" << std::to_string(this->negative_slope) << ", " << x.actual_size << ");" << std::endl;
  result = ss.str();

  return ge::SUCCESS;
}

Status LeakyReluApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("negative_slope", this->negative_slope),
                          "Failed to get LeakyRelu negative_slope attr, node = %s", node->GetNamePtr());
  GELOGI("name:%s, negative_slope:%f", node->GetNamePtr(), this->negative_slope);
  return ge::SUCCESS;
}

static ApiCallRegister<LeakyReluApiCall> register_leaky_relu_api_call("LeakyReluApiCall");
}  // namespace codegen