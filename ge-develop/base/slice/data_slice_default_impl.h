/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLICE_DATASLICE_DATA_SLICE_DEFAULT_IMPL_H_
#define SLICE_DATASLICE_DATA_SLICE_DEFAULT_IMPL_H_

#include "slice/data_slice_infer_base.h"
namespace ge {
// Default
class DataSliceDefaultImpl : public DataSliceInferBase {
 public:
  DataSliceDefaultImpl() {}
  virtual ~DataSliceDefaultImpl() = default;
  Status InferAxisSlice(ge::Operator &op, const AxisTypeInfo &slice_info,
    const DataSliceType &out_data_slice, DataSliceType &in_data_slice) override;
};
}
#endif // SLICE_DATASLICE_DATA_SLICE_DEFAULT_IMPL_H_
