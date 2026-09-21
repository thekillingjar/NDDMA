/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_COMMON_BASE_TEMPLATE_H
#define OPTIMIZE_PLATFORM_COMMON_BASE_TEMPLATE_H

#include "ascendc_ir.h"
#include "optimize/schedule_utils.h"
#include "common/common_utils.h"

namespace optimize {

enum class GenerationMode : std::uint32_t {
  kBaseCase = 0U,  // 基础模式，仅基于通用模板生成新的模板
  kAppendCase,     // 叠加模式，同时基于通用模板和已经生成的模板生成
};

class BaseTemplate {
 public:
  virtual ~BaseTemplate() = default;
  explicit BaseTemplate() = default;

  virtual std::string GenName(const std::string &general_case_name) = 0;

  virtual GenerationMode GetGenerationMode() {
    return GenerationMode::kBaseCase;
  }

  virtual ge::Status Generate(const ge::AscGraph &origin_graph, const ge::AscGraph &based_case,
                              ge::AscGraph &new_case) = 0;

  virtual bool NeedDropBasedCase(const ge::AscGraph &origin_graph, const ge::AscGraph &based_case,
                                 const ge::AscGraph &new_case) = 0;

  virtual std::string GetScoreFunc([[maybe_unused]]const ge::AscGraph &origin_graph, [[maybe_unused]]const ge::AscGraph &base_graph) {
    return "";
  }

  BaseTemplate(const BaseTemplate &) = delete;
  BaseTemplate &operator=(const BaseTemplate &) = delete;
  BaseTemplate(BaseTemplate &&) = delete;
  BaseTemplate &operator=(BaseTemplate &&) = delete;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_COMMON_BASE_TEMPLATE_H
