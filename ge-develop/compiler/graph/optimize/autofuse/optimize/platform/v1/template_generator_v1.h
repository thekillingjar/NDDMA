/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V1_TEMPLATE_GENERATOR_V1_H
#define OPTIMIZE_PLATFORM_V1_TEMPLATE_GENERATOR_V1_H

#include "platform/common/base_template_generator.h"
#include "platform/v1/template/unaligned_template.h"
#include "platform/v1/template/brc_inline_template.h"

namespace optimize {
class TemplateGeneratorV1 : public BaseTemplateGenerator {
 public:
  ~TemplateGeneratorV1() override = default;
  explicit TemplateGeneratorV1() {
    strategies_.push_back(ge::ComGraphMakeUnique<UnalignedTemplate>());
    strategies_.push_back(ge::ComGraphMakeUnique<BrcInlineTemplate>());
  };

  TemplateGeneratorV1(const TemplateGeneratorV1 &) = delete;
  TemplateGeneratorV1 &operator=(const TemplateGeneratorV1 &) = delete;
  TemplateGeneratorV1(TemplateGeneratorV1 &&) = delete;
  TemplateGeneratorV1 &operator=(TemplateGeneratorV1 &&) = delete;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_V1_TEMPLATE_GENERATOR_V1_H
