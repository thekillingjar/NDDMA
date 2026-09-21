/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_LABEL_MAKER_H_
#define GE_GRAPH_PASSES_LABEL_MAKER_H_

#include <vector>

#include "graph/node.h"
#include "graph/label/label_maker_factory.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
class LabelMaker {
 public:
  LabelMaker(const ComputeGraphPtr &graph, const NodePtr &owner) : parent_node_(owner), parent_graph_(graph) {}

  virtual ~LabelMaker() {
    parent_node_ = nullptr;
    parent_graph_ = nullptr;
  }

  virtual Status Run(uint32_t &label_index) = 0;

  NodePtr AddStreamActive(const ComputeGraphPtr &graph, const std::string &name);
  std::vector<std::string> GetActiveLabelList(const ComputeGraphPtr &graph) const;

  NodePtr AddLabelSetEnter(const ComputeGraphPtr &graph, const std::string &name, uint32_t index,
                           const NodePtr &stream_active) const;
  NodePtr AddLabelSetLeave(const ComputeGraphPtr &graph, const std::string &name, uint32_t index);

  NodePtr AddLabelGotoEnter(const ComputeGraphPtr &graph, const std::string &name, uint32_t index) const;
  NodePtr AddLabelGotoLeave(const ComputeGraphPtr &graph, const std::string &name, uint32_t index);

  NodePtr AddLabelSwitchEnter(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                              const std::vector<uint32_t> &labels) const;
  NodePtr AddLabelSwitchLeave(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                              const std::vector<uint32_t> &labels) const;

  NodePtr AddLabelSwitchIndex(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                              const NodePtr &sw_node, uint32_t parent_index) const;

  LabelMaker &operator=(const LabelMaker &model) = delete;
  LabelMaker(const LabelMaker &model) = delete;

 protected:
  NodePtr parent_node_;
  ComputeGraphPtr parent_graph_;

 private:
  void LinkToGraphHead(const ComputeGraphPtr &graph, const NodePtr &node) const;
  void LinkToGraphTail(const ComputeGraphPtr &graph, const NodePtr &node) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_LABEL_MAKER_H_
