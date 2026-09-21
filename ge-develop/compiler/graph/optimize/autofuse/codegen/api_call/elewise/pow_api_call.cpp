/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pow_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils//asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"

namespace codegen {
using namespace std;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;

Status PowApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                            const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                            std::string &result) const {
  const auto &x1 = inputs[0].get();
  const auto &x2 = inputs[1].get();
  const auto &y = outputs[0].get();

  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  GE_ASSERT_TRUE(it != this->tmp_buf_id.end(), "PowApiCall cannot find tmp buffer id to use.");
  id = it->second;

  GELOGD("x1, is_constant:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d",
         static_cast<int32_t>(x1.is_constant), static_cast<int32_t>(x1.is_ub_scalar),
         static_cast<int32_t>(x1.need_gen_get_value_of_ub_scalar));
  GELOGD("x2, is_constant:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d",
         static_cast<int32_t>(x2.is_constant), static_cast<int32_t>(x2.is_ub_scalar),
         static_cast<int32_t>(x2.need_gen_get_value_of_ub_scalar));
  stringstream ss;
  const bool x1_is_scalar_scene = (x1.is_constant) || (x1.is_ub_scalar && x1.need_gen_get_value_of_ub_scalar);
  const bool x2_is_scalar_scene = (x2.is_constant) || (x2.is_ub_scalar && x2.need_gen_get_value_of_ub_scalar);
  
  // api层面x1 x2 dtype类型一样
  std::string x1_dtype_name;
  std::string x2_dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x1.dtype, x1_dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x1.dtype));
  GE_CHK_STATUS_RET(Tensor::DtypeName(x2.dtype, x2_dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x2.dtype));
  GE_ASSERT_TRUE(x1_dtype_name == x2_dtype_name, "x1_dtype_name:%s, x2_dtype_name:%s", x1_dtype_name.c_str(),
                 x2_dtype_name.c_str());

  std::string x1_scalar =
      x1.need_gen_get_value_of_ub_scalar ? ("(" + x1_dtype_name + ")" + x1.ub_scalar_name) : x1.Str();
  std::string x2_scalar =
      x2.need_gen_get_value_of_ub_scalar ? ("(" + x2_dtype_name + ")" + x2.ub_scalar_name) : x2.Str();

  ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], ";  // 输出
  if (x1_is_scalar_scene && x2_is_scalar_scene) {
    ss << x1_scalar << ", " << x2_scalar << ", " << y.actual_size << ", " << tpipe.tmp_buf << "_" << std::to_string(id)
       << ");" << std::endl;
  } else {
    if (x1_is_scalar_scene) {
      ss << x1_scalar << ", ";  // 输入1
    } else {
      ss << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], ";  // 输入1
    }
    if (x2_is_scalar_scene) {
      ss << x2_scalar << ", ";  // 输入2
    } else {
      ss << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], ";  // 输入2
    }
    if (x1_is_scalar_scene) {
      // x1为scalar, cnt为x2.actual_size
      ss << x2.actual_size << ", " << tpipe.tmp_buf << "_" << std::to_string(id) << ");" << std::endl;
    } else {
      // x2为scalar或都为tensor的场景, cnt以x1.actual_size为准
      ss << x1.actual_size << ", " << tpipe.tmp_buf << "_" << std::to_string(id) << ");" << std::endl;
    }
  }
  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<PowApiCall> register_pow_api_call("PowApiCall");
}  // namespace codegen
