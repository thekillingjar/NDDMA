/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slice/data_slice_default_impl.h"
#include "slice/data_slice_toolkit.h"
#include "slice/data_slice_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/type_utils.h"

namespace ge {
static AxisInferRegister registerDefault(ge::AxisType::UNSPLIT,
  [] (void) noexcept ->DataSliceInferBase* {return new (std::nothrow) DataSliceDefaultImpl();});

// Default
Status DataSliceDefaultImpl::InferAxisSlice(ge::Operator &op, const AxisTypeInfo &slice_info,
    const DataSliceType &out_data_slice, DataSliceType &in_data_slice)
{
  GELOGI("Default infer func, op:%s, type:%s, axis type:%d",
         DataSliceGetName(op).c_str(), DataSliceGetOpType(op).c_str(), static_cast<int8_t>(slice_info.GetAxisType()));
  in_data_slice = out_data_slice;
  return SUCCESS;
}
}
