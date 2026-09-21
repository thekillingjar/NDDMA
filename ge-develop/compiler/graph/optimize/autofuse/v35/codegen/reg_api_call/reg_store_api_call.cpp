/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_store_api_call.h"

#include <sstream>

#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "reg_api_call_utils.h"
#include "api_call/utils/api_call_factory.h"

using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;
namespace {
constexpr uint64_t kDmaMaxLen = 2U;
}

namespace codegen {
Status StoreRegApiCall::ParseAttr(const ascir::NodeView &node) {
  // 存在多个Store写同一个Tensor不同offset的场景, repeats用当前Store节点的
  repeats_ = node->outputs[0U].attr.repeats;
  (void)node->attr.ir_attr->GetAttrValue("offset", offset_);
  return ge::SUCCESS;
}

Status StoreRegApiCall::PreProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                   const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                   std::string &result) const {
  GE_ASSERT_TRUE(!outputs.empty());
  const_cast<Tensor &>(outputs.front().get()).axis_size = repeats_;
  GE_ASSERT_SUCCESS(ApiCall::PreProcess(tpipe, current_axis, outputs, result));
  return ge::SUCCESS;
}

Status StoreRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                 const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                 const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                 std::string &result) const {
  std::stringstream ss;
  const auto &gm = outputs[0].get();
  const auto &ub = inputs[0].get();
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    std::string dtype_name;
    Tensor::DtypeName(gm.dtype, dtype_name);
    ss << "DataCopyPadExtend<" << dtype_name << ", AscendC::PaddingMode::Normal>(" << gm << "[offset], " << ub << ", "
       << "curAivM, curAivN, 0, (shapeN - curAivN));" << std::endl;
  } else {
    DataCopyParams param;

    bool status = CalculateDmaParams(tpipe, gm, gm, param);
    GE_ASSERT_TRUE(status, "CalculateDmaParams failed");
    std::string gm_offset = tpipe.tiler.Offset(current_axis, gm.axis, gm.axis_strides);
    CreateEnhanceDmaCall(tpipe, ub, gm, gm_offset, param, offset_, ss, false);

    if (IsUnitLastRead(*(this->inputs[0])) && ub.is_load_link_store_and_vec) {
      std::string offset = offset_.Str().get();
      offset = GenValidName(offset);
      GenerateLinkStoreEventCode(ub, offset, ss);
    }
  }

  result = ss.str();
  return ge::SUCCESS;
}
static ApiCallRegister<StoreRegApiCall> register_store_reg_api_call("StoreRegApiCall");
}  // namespace codegen