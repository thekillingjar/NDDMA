/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_COMMON_TEMPLATE_BRC_INLINE_TEMPLATE_V2_H
#define OPTIMIZE_PLATFORM_COMMON_TEMPLATE_BRC_INLINE_TEMPLATE_V2_H

#include "platform/v1/template/brc_inline_template.h"

namespace optimize {
class BrcInlineTemplateV2 : public BrcInlineTemplate {
 public:
  ~BrcInlineTemplateV2() override = default;
  explicit BrcInlineTemplateV2() = default;

  static bool IsNodeSupportBrcInline(const ge::NodePtr &node);
  static ge::Status AlignTensor(const ge::NodePtr &node, const ge::AscTensor *tensor);
  ge::Status AlignAssociateNodes(const ge::AscGraph &graph, const ge::AscNodePtr &brc_node);

  GenerationMode GetGenerationMode() override;
  ge::Status Generate(const ge::AscGraph &origin_graph, const ge::AscGraph &based_case,
                      ge::AscGraph &new_case) override;
  bool NeedDropBasedCase(const ge::AscGraph &origin_graph, const ge::AscGraph &based_case,
                         const ge::AscGraph &new_case) override;

  BrcInlineTemplateV2(const BrcInlineTemplateV2 &) = delete;
  BrcInlineTemplateV2 &operator=(const BrcInlineTemplateV2 &) = delete;
  BrcInlineTemplateV2(BrcInlineTemplateV2 &&) = delete;
  BrcInlineTemplateV2 &operator=(BrcInlineTemplateV2 &&) = delete;

private:
  bool IsNodeAligned(const ge::NodePtr &node) const;
  void MarkNodeAligned(const ge::NodePtr &node);

  bool IsNodeAddedRemovePad(const ge::NodePtr &node) const;
  void MarkNodeAddedRemovePad(const ge::NodePtr &node);

  std::set<ge::NodePtr> aligned_nodes_;
  std::set<ge::NodePtr> add_remove_pad_nodes_;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_COMMON_TEMPLATE_BRC_INLINE_TEMPLATE_V2_H