/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_MICRO_LEAKY_RELU_API_CALL_H__
#define __AUTOFUSE_MICRO_LEAKY_RELU_API_CALL_H__

#include "micro_api_call.h"

namespace codegen {
class MicroLeakyReluApiCall final : public MicroApiCall {
 public:
  explicit MicroLeakyReluApiCall(const std::string &api_name) : MicroApiCall(api_name) {}
  ~MicroLeakyReluApiCall() override = default;
  Status Init(const ascir::NodeView &node) override;
  Status Generate(const TensorManager &tensor_mng, const TPipe &tpipe, CallParam &param, std::string &result) override;

 private:
  float negative_slope_ = 0.0f;
};
}  // namespace codegen

#endif  // __AUTOFUSE_MICRO_LEAKY_RELU_API_CALL_H__