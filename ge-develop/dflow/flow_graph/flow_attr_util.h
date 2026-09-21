/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_GRAPH_FLOW_ATTR_UTIL_H_
#define FLOW_GRAPH_FLOW_ATTR_UTIL_H_

#include <vector>
#include <map>
#include <cstdint>
#include "flow_graph/flow_attr.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
namespace dflow {
class FlowAttrUtil {
public:
  static graphStatus SetAttrsToTensorDesc(const std::vector<DataFlowInputAttr> &attrs, GeTensorDescPtr &tensor_desc);

private:
  static bool CheckAttrsIsSupport(const std::vector<DataFlowInputAttr> &attrs);
  static graphStatus SetCountBatchAttr(const void *const attr_value, GeTensorDescPtr &tensor_desc);
  static graphStatus SetTimeBatchAttr(const void *const attr_value, GeTensorDescPtr &tensor_desc);
  using SetAttrFunc = graphStatus (*)(const void *const attr_value, GeTensorDescPtr &input_tensor_desc);
  static const std::map<DataFlowAttrType, SetAttrFunc> set_attr_funcs_;
};
}  // namespace dflow
}  // namespace ge
#endif // FLOW_GRAPH_FLOW_ATTR_UTIL_H_
