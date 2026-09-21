/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_IF_OP_LABEL_PASS_H_
#define GE_GRAPH_PASSES_IF_OP_LABEL_PASS_H_

#include "graph/node.h"
#include "graph/label/label_maker.h"
/*******************************************************************************
                                                                +------------+
                                                                |    Node    |
                                                                +------------+
                                                                |    Node    |
                                                                +------------+
                                                                |     If     |
                                                                +------------+
              +-----------+
              |   Node    |                                     +------------+
              +-----------+                                    /|SwitchByIdx |
              |   Node    |                                   A +------------+
              +-----------+                                  / \|LabelSet(1) |
              |    If     |                                 |   +------------+
              +-----------+                                 |   |StreamActive|
              |   Node    |                                 |   +------------+
              +-----------+                                 |   |     t      |
              |   Node    |                                 |   +------------+
              +-----------+                                 |   |     h      |
              |   Node    |                                 |   +------------+
              +-----------+                                 |   |     e      |
              |   Node    |                                 |   +------------+
              +-----------+                                 |   |     n      |
                                                            |   +------------+
                                               ====>        |   | LabelGoto  |\
                                                            V   +------------+ \
    +-----------+      +-----------+                         \                  \
    |     t     |      |     e     |                          \ +------------+   |
    +-----------+      +-----------+                           \|LabelSet(0) |   |
    |     h     |      |     l     |                            +------------+   |
    +-----------+      +-----------+                            |StreamActive|   |
    |     e     |      |     s     |                            +------------+   |
    +-----------+      +-----------+                            |     e      |   |
    |     n     |      |     e     |                            +------------+   |
    +-----------+      +-----------+                            |     l      |   |
                                                                +------------+   |
                                                                |     s      |   |
                                                                +------------+   V
                                                                |     e      |  /
                                                                +------------+ /
                                                                |  LabelSet  |/
                                                                +------------+

                                                                +------------+
                                                                |    Node    |
                                                                +------------+
                                                                |    Node    |
                                                                +------------+
                                                                |    Node    |
                                                                +------------+
*******************************************************************************/
namespace ge {
class IfOpLabelMaker : public LabelMaker {
 public:
  IfOpLabelMaker(const ComputeGraphPtr &graph, const NodePtr &owner) : LabelMaker(graph, owner) {}

  ~IfOpLabelMaker() override {}

  virtual Status Run(uint32_t &label_index);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_IF_OP_LABEL_PASS_H_
