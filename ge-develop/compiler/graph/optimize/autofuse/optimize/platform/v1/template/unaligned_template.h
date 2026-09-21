/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_COMMON_TEMPLATE_UNALIGN_TEMPLATE_H
#define OPTIMIZE_PLATFORM_COMMON_TEMPLATE_UNALIGN_TEMPLATE_H

#include "platform/common/base_template.h"

namespace optimize {
class UnalignedTemplate : public BaseTemplate {
 public:
  ~UnalignedTemplate() override = default;
  explicit UnalignedTemplate() = default;

  std::string GenName(const std::string &general_case_name) override;
  static bool NeedRemovePad(const ge::AscNodePtr &node);
  static ge::Status UnAlignVectorizedStrides(const ge::AscNodePtr &node);
  static ge::Status ReverseDfsUnAlignNode(ge::AscGraph &impl_graph, const ge::NodePtr &ge_node,
                                          std::set<ge::NodePtr> &visited_nodes);
  ge::Status Generate(const ge::AscGraph &origin_graph, const ge::AscGraph &based_case,
                      ge::AscGraph &new_case) override;
  bool NeedDropBasedCase(const ge::AscGraph &origin_graph, const ge::AscGraph &based_case,
                         const ge::AscGraph &new_case) override;

  UnalignedTemplate(const UnalignedTemplate &) = delete;
  UnalignedTemplate &operator=(const UnalignedTemplate &) = delete;
  UnalignedTemplate(UnalignedTemplate &&) = delete;
  UnalignedTemplate &operator=(UnalignedTemplate &&) = delete;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_COMMON_TEMPLATE_UNALIGN_TEMPLATE_H