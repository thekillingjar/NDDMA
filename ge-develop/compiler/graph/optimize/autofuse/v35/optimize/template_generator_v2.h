/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_TEMPLATE_GENERATOR_V2_H
#define OPTIMIZE_PLATFORM_V2_TEMPLATE_GENERATOR_V2_H

#include "platform/common/base_template_generator.h"
#include "v35/optimize/template/brc_inline_template_v2.h"
#include "v35/optimize/template/nddma_template.h"
#include "v35/optimize/template/load_to_nddma_template.h"

namespace optimize {
class TemplateGeneratorV2 : public BaseTemplateGenerator {
 public:
  ~TemplateGeneratorV2() override = default;
  explicit TemplateGeneratorV2() {
    strategies_.push_back(ge::ComGraphMakeUnique<NddmaTemplate>());
    strategies_.push_back(ge::ComGraphMakeUnique<LoadToNddmaTemplate>());
    // 临时关闭 V2 的 brc inline 模板  // strategies_.push_back(ge::ComGraphMakeUnique<BrcInlineTemplateV2>());
  };

  TemplateGeneratorV2(const TemplateGeneratorV2 &) = delete;
  TemplateGeneratorV2 &operator=(const TemplateGeneratorV2 &) = delete;
  TemplateGeneratorV2(TemplateGeneratorV2 &&) = delete;
  TemplateGeneratorV2 &operator=(TemplateGeneratorV2 &&) = delete;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_V2_TEMPLATE_GENERATOR_V2_H