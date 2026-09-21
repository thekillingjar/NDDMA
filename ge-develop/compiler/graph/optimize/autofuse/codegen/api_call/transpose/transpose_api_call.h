/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_TRANSPOSE__API_CALL_H__
#define __AUTOFUSE_TRANSPOSE__API_CALL_H__
#include "codegen_kernel.h"
#include "transpose_base_type.h"

namespace codegen {
class TransposeApiCall final : public ApiCall {
public:
  using ApiCall::Generate;
  explicit TransposeApiCall(const std::string &api_name) : ApiCall(api_name) {}
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
  Status ParseAttr(const ascir::NodeView &node) override;
  Status CodeGenGetTransposeType(const Tensor &inputs, const Tensor &outputs, AutoFuseTransposeType &transpose_type) const;
  ~TransposeApiCall() final = default;
private:
    std::string host_api_tiling_data_type; // = GetApiTilingTypeName(); // "tran0_tiling_data";
    std::string device_api_tiling_data_type; // = GetApiTilingTypeName(); // "tran0_tiling_data";
    std::string api_tiling_data_field; // = GetApiTilingFieldName(); // "tran0_tiling_data";
};
}
#endif // __AUTOFUSE_TRANSPOSE__API_CALL_H__