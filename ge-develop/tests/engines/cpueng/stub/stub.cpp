/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stub.h"
#include "common/aicore_util_types.h"

using namespace aicpu;
using namespace ge;
using namespace std;

RunContext CreateContext()
{
    RunContext context;
    context.dataMemSize = 10000;
    context.dataMemBase = (uint8_t *) (intptr_t) 1000;
    context.sessionId = 10000011011;
    return context;
}

void DestroyContext(RunContext& context) {}

TestForGetSoPath &TestForGetSoPath::Instance()
{
    static TestForGetSoPath instance;
    return instance;
}

bool IsPathExist(const std::string &path)
{
  return true;
}

namespace fe {
class OpSliceUtil {
 public:
  explicit OpSliceUtil();
  virtual ~OpSliceUtil();
  static Status SetOpSliceInfo(const ge::NodePtr &node, const SlicePattern &slice_pattern,
                               const bool &support_stride_write);
};

Status OpSliceUtil::SetOpSliceInfo(const ge::NodePtr &node,
                      const SlicePattern &slice_pattern,
                      const bool &support_stride_write) {
  if (node == nullptr) {
    return FAILED;
  }
  return SUCCESS;
}
};

namespace ge {
graphStatus HostCpuTestOp::Compute(Operator &op, const std::map<std::string,
  const Tensor> &inputs, std::map<std::string, Tensor> &outputs) {
    return SUCCESS;
}
}