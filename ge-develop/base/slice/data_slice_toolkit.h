/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLICE_DATASLICE_DATA_SLICE_TOOLKIT_H_
#define SLICE_DATASLICE_DATA_SLICE_TOOLKIT_H_

#include <string>
#include "graph/ge_error_codes.h"
#include "ge/ge_api_types.h"
#include "graph/operator.h"
namespace ge {
template <typename T>
std::string DataSliceGetName(const T& op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetName(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    return "None";
  }
  return op_ascend_name.GetString();
}

template <typename T>
std::string DataSliceGetOpType(const T& op) {
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetOpType(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    return "None";
  }
  return op_ascend_name.GetString();
}
}  // namespace ge
#endif // SLICE_DATASLICE_DATA_SLICE_TOOLKIT_H_
