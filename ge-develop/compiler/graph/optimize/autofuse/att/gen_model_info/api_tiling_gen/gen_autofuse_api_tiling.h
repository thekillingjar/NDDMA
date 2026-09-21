/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_GEN_AUTOFUSE_API_TILING_H
#define AUTOFUSE_GEN_AUTOFUSE_API_TILING_H

#include <string>
#include <map>
#include <utility>
#include <cstdint>
#include "base/base_types.h"
#include "common/checker.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace att {
class AutofuseApiTilingGenerator {
 public:
  AutofuseApiTilingGenerator(const ge::AscGraph &graph, const ge::AscNodePtr &node, std::string tiling_data_type,
                             uint32_t tiling_case_id)
      : tiling_case_id_(tiling_case_id), graph_(graph), node_(node), tiling_data_type_(std::move(tiling_data_type)) {}
  ~AutofuseApiTilingGenerator() = default;
  ge::Status Generate();
  std::string GetFuncImpl() const {
    return function_impl_;
  }
  std::string GetFuncInvoke() const {
    return function_invoke_;
  }
  std::string GetHeadFiles() const {
    return head_files_;
  }

 private:
  uint32_t tiling_case_id_;
  ge::AscGraph graph_;
  ge::AscNodePtr node_;
  std::string tiling_data_type_;
  std::string function_impl_;
  std::string function_invoke_;
  std::string head_files_;
};
}  // namespace att

#endif  // AUTOFUSE_GEN_AUTOFUSE_API_TILING_H
