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
#include "graph/debug/ge_attr_define.h"
#include "ge_common/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "common/checker.h"
#include "graph/utils/tensor_utils.h"
#include "math_util.h"
#include "base/err_msg.h"

namespace ge {
Status ContinuousOutputNodeChecker(const NodeCheckerParam &param) {
  int64_t expect_offset = -1;
  for (const auto &out_data_anchor : param.node->GetAllOutDataAnchorsPtr()) {
    const auto out_index = out_data_anchor->GetIdx();
    const auto out_tensor = param.node->GetOpDescBarePtr()->GetOutputDescPtr(out_index);
    GE_ASSERT_NOTNULL(out_tensor);
    int64_t size = -1;
    GE_ASSERT_SUCCESS(TensorUtils::GetSize(*out_tensor, size), "node: %s, out_index: %d", param.node->GetNamePtr(),
                      out_index);
    GE_ASSERT_SUCCESS(NodeCheckerUtils::AlignMemSize(size), "size: %" PRId64 " < 0, node: %s, out_index: %d", size,
                      param.node->GetNamePtr(), out_index);

    int64_t offset = 0;
    GE_ASSERT_SUCCESS(NodeCheckerUtils::GetOutputOffset(param.node->GetOpDescBarePtr(), out_index, offset));
    if (expect_offset == -1) {
      GE_ASSERT_TRUE(!AddOverflow(offset, size, expect_offset), "offset: %" PRId64 ", size: %" PRId64 "", offset, size);
      continue;
    }
    if (expect_offset != offset) {
      REPORT_INNER_ERR_MSG("E19999", "continuous output node memory check failed. node: %s, out_index: %d,"
          " offset: %" PRId64 ", expect_offset: %" PRId64 "", NodeCheckerUtils::NodeName(param.node).c_str(), out_index,
          offset, expect_offset);
      GELOGE(FAILED, "continuous output node memory check failed. node: %s, out_index: %d, offset: %" PRId64 ", "
          "expect_offset: %" PRId64 "", NodeCheckerUtils::NodeName(param.node).c_str(), out_index, offset,
          expect_offset);
      GE_ASSERT_SUCCESS(NodeCheckerUtils::ErrorLogAllOutputs(param));
      return FAILED;
    }
    GE_ASSERT_TRUE(!AddOverflow(offset, size, expect_offset), "offset: %" PRId64 ", size: %" PRId64 "", offset, size);
  }
  return SUCCESS;
}

REGISTER_SPECIAL_NODE_CHECKER(kSpecialNodeTypeContinuousOutput, ContinuousOutputNodeChecker);
}  // namespace ge
