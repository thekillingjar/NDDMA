/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_REG_REDUCE_API_CALL_H__
#define __AUTOFUSE_REG_REDUCE_API_CALL_H__
#include "codegen_kernel.h"

namespace codegen {
class RegReduceApiCall : public ApiCall {
public:
  using ApiCall::Generate;
  explicit RegReduceApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegReduceApiCall() final = default;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
  Status CallParseAttr(const ascir::NodeView &node) {
      return ParseAttr(node);
  }
protected:
    Status ParseAttr(const ascir::NodeView &node) override;

private:
    std::string is_reuse_source_ = "false";
};
}
#endif // __AUTOFUSE_REG_REDUCE_API_CALL_H__