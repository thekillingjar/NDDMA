/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/common/node_optimize_utils.h"

namespace fe {
Status NodeOptimizeUtils::GetPreNode(const ge::NodePtr &node_ptr, const int &index, ge::NodePtr &pre_node_ptr) {
  ge::InDataAnchorPtr in_data_anchor = node_ptr->GetInDataAnchor(index);
  FE_CHECK_NOTNULL(in_data_anchor);
  ge::OutDataAnchorPtr peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(peer_out_data_anchor);
  pre_node_ptr = peer_out_data_anchor->GetOwnerNode();
  FE_CHECK_NOTNULL(pre_node_ptr);
  return SUCCESS;
}
}  // namespace fe
