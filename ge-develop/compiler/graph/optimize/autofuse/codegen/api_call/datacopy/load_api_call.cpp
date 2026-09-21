/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>

#include "ascir_ops.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "../utils/api_call_factory.h"
#include "../utils/api_call_utils.h"

#include "load_api_call.h"

using namespace ge::ops;
using namespace ge::ascir_op;

namespace {
constexpr uint64_t kDmaMaxLen = 2U;
}
namespace codegen {
Status LoadApiCall::ParseAttr(const ascir::NodeView &node) {
  (void)node->attr.ir_attr->GetAttrValue("offset", offset_);
  return ge::SUCCESS;
}

Status LoadApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                             const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                             const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                             std::string &result) const {
  std::stringstream ss;
  const auto &gm = inputs[0].get();
  const auto &ub = outputs[0].get();
  DataCopyParams param;
  (void)CalculateDmaParams(tpipe, ub, ub, param);
  std::string gm_offset = ub.is_ub_scalar ? "0" : tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides);
  if (param.repeats.size() <= kDmaMaxLen) {
    DmaParams dma_param;
    SetDmaParams(tpipe, param, dma_param, true);
    ss << "DataCopyPadExtend(" << ub << ", " << gm << "[" << gm_offset << " + " << tpipe.tiler.Size(offset_) << "], "
       << dma_param.block_count << ", " << dma_param.block_len << ", " << dma_param.src_stride << ", "
       << dma_param.dst_stride << ");" << std::endl;
  } else {
    CreateDmaCall(tpipe, gm, ub, gm_offset, param, offset_, ss, true);
  }

  result = ss.str();
  return ge::SUCCESS;
}
static ApiCallRegister<LoadApiCall> register_load_api_call("LoadApiCall");
}  // namespace codegen