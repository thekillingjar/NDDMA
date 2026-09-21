/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_ANCHOR_UTILS_H_
#define INC_GRAPH_UTILS_ANCHOR_UTILS_H_

#include "graph/anchor.h"
#include "graph/node.h"

namespace ge {
class AnchorUtils {
 public:
  // Get anchor status
  static AnchorStatus GetStatus(const DataAnchorPtr &data_anchor);

  // Set anchor status
  static graphStatus SetStatus(const DataAnchorPtr &data_anchor, const AnchorStatus anchor_status);

  static int32_t GetIdx(const AnchorPtr &anchor);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_ANCHOR_UTILS_H_
