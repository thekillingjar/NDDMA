/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLICE_DATASLICE_DATA_SLICE_INFER_BASE_H_
#define SLICE_DATASLICE_DATA_SLICE_INFER_BASE_H_

#include <vector>
#include "graph/operator.h"
#include "ge/ge_api_types.h"
#include "graph/axis_type_info.h"
namespace ge {
using DataSliceType = std::vector<std::vector<std::vector<int64_t>>>;
class DataSliceInferBase {
 public:
  DataSliceInferBase() = default;
  virtual ~DataSliceInferBase() = default;
  virtual Status InferAxisSlice(ge::Operator &op, const AxisTypeInfo &slice_info,
    const DataSliceType &out_data_slice, DataSliceType &in_data_slice) = 0;
};
}
#endif // SLICE_DATASLICE_DATA_SLICE_INFER_BASE_H_
