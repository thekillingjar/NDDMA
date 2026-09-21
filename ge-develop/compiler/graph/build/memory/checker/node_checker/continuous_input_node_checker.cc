/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../node_checker_register.h"
#include "../node_checker_utils.h"
#include "ge_common/ge_api_error_codes.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "common/checker.h"
#include "graph/utils/tensor_utils.h"
#include "math_util.h"
#include "base/err_msg.h"

namespace ge {
Status ContinuousInputNodeChecker(const NodeCheckerParam &param) {
  int64_t expect_offset = -1;
  for (const auto &in_data_anchor : param.node->GetAllInDataAnchorsPtr()) {
    const auto &peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_data_anchor);
    const auto peer_node = peer_out_data_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_node);
    GE_ASSERT_NOTNULL(peer_node->GetOpDescBarePtr());
    const auto out_index = peer_out_data_anchor->GetIdx();
    const auto out_tensor = peer_node->GetOpDescBarePtr()->GetOutputDescPtr(out_index);
    GE_ASSERT_NOTNULL(out_tensor);

    int64_t stride = 0;
    GE_ASSERT_SUCCESS(NodeCheckerUtils::GetStrideForContinuousInput(peer_node, out_index, stride),
                      "peer_node: %s, out_index: %d", NodeCheckerUtils::NodeName(peer_node).c_str(), out_index);

    int64_t offset = 0;
    GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(peer_node->GetOpDescBarePtr(), out_index, offset));
    if (expect_offset == -1) {
      GE_ASSERT_TRUE(!AddOverflow(offset, stride, expect_offset), "offset: %" PRId64 ", stride: %" PRId64 "", offset,
                     stride);
      continue;
    }

    if (expect_offset != offset) {
      REPORT_INNER_ERR_MSG("E19999", "continuous input node memory check failed. node: %s, input_index: %d,"
          " input node %s, out_index: %d, offset: %" PRId64 ", expect_offset: %" PRId64 "",
          NodeCheckerUtils::NodeName(param.node).c_str(), in_data_anchor->GetIdx(),
          NodeCheckerUtils::NodeName(peer_node).c_str(), out_index, offset, expect_offset);
      GELOGE(FAILED, "continuous input node memory check failed. node: %s, input_index: %d, input node %s,"
          " out_index: %d, offset: %" PRId64 ", expect_offset: %" PRId64 "",
          NodeCheckerUtils::NodeName(param.node).c_str(), in_data_anchor->GetIdx(),
          NodeCheckerUtils::NodeName(peer_node).c_str(), out_index, offset, expect_offset);
      GE_ASSERT_SUCCESS(NodeCheckerUtils::ErrorLogAllInputs(param));
      return FAILED;
    }
    GE_ASSERT_TRUE(!AddOverflow(offset, stride, expect_offset), "offset: %" PRId64 ", stride: %" PRId64 "", offset,
                   stride);
  }
  return SUCCESS;
}

REGISTER_SPECIAL_NODE_CHECKER(kSpecialNodeTypeContinuousInput, ContinuousInputNodeChecker);
}  // namespace ge
