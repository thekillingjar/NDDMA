/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLICE_DATASLICE_DATA_SLICE_HELPER_H_
#define SLICE_DATASLICE_DATA_SLICE_HELPER_H_

#include <vector>
#include "ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/axis_type_info.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
namespace ge {
class DataSliceHelper {
 public:
  static Status InferAxisSlice(OpDescPtr &op, const AxisTypeInfo &slice_info);
  static Status GetSliceInfo(OpDescPtr &op, std::vector<AxisTypeInfo> &axis_type_vec);
  static Status GetSliceInfo(const NodePtr &node, std::vector<AxisTypeInfo> &axis_type_vec);
  static Status InferDavinciAxisSlice(OpDescPtr &op, const AxisTypeInfo &slice_info);
  static Status GetDavinciSliceInfo(const NodePtr &node, std::vector<AxisTypeInfo> &axis_type_vec);
 private:
  // cut tensor: axis index: shape range
  using DataSliceType = std::vector<std::vector<std::vector<int64_t>>>;
  static Status SetInputSlice(OpDescPtr &op, const AxisTypeInfo &slice_info, DataSliceType &input_slice);
  static Status InferDavinciCommonOpSlice(OpDescPtr &op, const AxisTypeInfo &slice_info);
  static Status InferDavinciSpecialOpSlice(OpDescPtr &op, const AxisTypeInfo &slice_info,
    const InferAxisSliceFunc &node_slice_infer_ptr);
};
}
#endif // SLICE_DATASLICE_DATA_SLICE_HELPER_H_
