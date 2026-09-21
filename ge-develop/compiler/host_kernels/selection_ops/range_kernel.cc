/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "range_kernel.h"

#include <cfloat>
#include <memory>
#include <set>

#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kRangeInputNum = 3;
constexpr uint32_t kRangeDimNum = 0;
constexpr size_t kStartIndex = 0;
constexpr size_t kLimitIndex = 1;
constexpr size_t kDeltaIndex = 2;
const std::set<DataType> kRangeSupportedType = {DT_INT32, DT_FLOAT, DT_DOUBLE, DT_INT64};
}  // namespace

Status RangeKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                            std::vector<GeTensorPtr> &v_output) {
  GELOGD("RangeKernel in");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  Status ret = RangeCheck(input);
  if (ret != SUCCESS) {
    return ret;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    return MEMALLOC_FAILED;
  }

  ConstGeTensorPtr start = input.at(kStartIndex);
  ConstGeTensorPtr limit = input.at(kLimitIndex);
  ConstGeTensorPtr delta = input.at(kDeltaIndex);
  DataType data_type = delta->GetTensorDesc().GetDataType();
  if (data_type == DT_FLOAT) {
    if (GetRange(*reinterpret_cast<const float *>(start->GetData().data()),
                 *reinterpret_cast<const float *>(limit->GetData().data()),
                 *reinterpret_cast<const float *>(delta->GetData().data()), output_ptr) != SUCCESS) {
      return PARAM_INVALID;
    }
  } else if (data_type == DT_INT32) {
    if (GetRange(*reinterpret_cast<const int32_t *>(start->GetData().data()),
                 *reinterpret_cast<const int32_t *>(limit->GetData().data()),
                 *reinterpret_cast<const int32_t *>(delta->GetData().data()), output_ptr) != SUCCESS) {
      return PARAM_INVALID;
    }
  } else if (data_type == DT_INT64) {
    if (GetRange(*reinterpret_cast<const int64_t *>(start->GetData().data()),
                 *reinterpret_cast<const int64_t *>(limit->GetData().data()),
                 *reinterpret_cast<const int64_t *>(delta->GetData().data()), output_ptr) != SUCCESS) {
      return PARAM_INVALID;
    }
  } else if (data_type == DT_DOUBLE) {
    if (GetRange(*reinterpret_cast<const double *>(start->GetData().data()),
                 *reinterpret_cast<const double *>(limit->GetData().data()),
                 *reinterpret_cast<const double *>(delta->GetData().data()), output_ptr) != SUCCESS) {
      return PARAM_INVALID;
    }
  }

  output_ptr->MutableTensorDesc().SetDataType(data_type);
  v_output.push_back(output_ptr);
  return SUCCESS;
}

Status RangeKernel::RangeCheck(const std::vector<ConstGeTensorPtr> &input) const {
  // check input number
  if (input.size() != kRangeInputNum) {
    GELOGI("The number of input for Range must be %zu.", kRangeInputNum);
    return NOT_CHANGED;
  }

  ConstGeTensorPtr start = input.at(0);
  ConstGeTensorPtr limit = input.at(1);
  ConstGeTensorPtr delta = input.at(2);

  GE_CHECK_NOTNULL(start);
  GE_CHECK_NOTNULL(limit);
  GE_CHECK_NOTNULL(delta);
  // check whether there is data in Tensor
  if (start->GetData().size() == 0 || limit->GetData().size() == 0 || delta->GetData().size() == 0) {
    GELOGI("Check data size fail. start: %zu, limit: %zu, delta: %zu", start->GetData().size(), limit->GetData().size(),
           delta->GetData().size());
    return NOT_CHANGED;
  }

  // check whether the data types are the same
  DataType type = start->GetTensorDesc().GetDataType();
  if ((type != limit->GetTensorDesc().GetDataType()) || (type != delta->GetTensorDesc().GetDataType())) {
    GELOGI("Data type of inputs for Range not matched.");
    return NOT_CHANGED;
  }

  // check whether are all scalars
  size_t range_dim = static_cast<size_t>(kRangeDimNum);
  bool all_scalar = (start->GetTensorDesc().GetShape().GetDimNum() == range_dim) &&
                    (limit->GetTensorDesc().GetShape().GetDimNum() == range_dim) &&
                    (delta->GetTensorDesc().GetShape().GetDimNum() == range_dim);
  if (!all_scalar) {
    GELOGI("Inputs for Range are not all scalars.");
    return NOT_CHANGED;
  }

  // check if input data type is supported
  if (kRangeSupportedType.find(type) == kRangeSupportedType.end()) {
    GELOGI("Range does not support this Data type: %s", TypeUtils::DataTypeToSerialString(type).c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

template <typename T>
bool RangeKernel::IsZero(const T &element) const {
  return (element == 0);
}

template <>
bool RangeKernel::IsZero<float32_t>(const float32_t &element) const {
  return (fabs(element) < FLT_EPSILON);
}

template <>
bool RangeKernel::IsZero<float64_t>(const float64_t &element) const {
  return (fabs(element) < DBL_EPSILON);
}

template <typename T>
Status RangeKernel::GetRange(const T start, const T limit, const T delta, GeTensorPtr &output) const {
  // check whether start, limit, delta is valid
  if (IsZero(delta)) {
    GELOGE(PARAM_INVALID, "Requires delta != 0");
    return PARAM_INVALID;
  }
  if (start > limit && delta > 0) {
    GELOGE(PARAM_INVALID, "Requires start <= limit when delta > 0");
    return PARAM_INVALID;
  }
  if (start < limit && delta < 0) {
    GELOGE(PARAM_INVALID, "Requires start >= limit when delta < 0");
    return PARAM_INVALID;
  }

  int64_t size = static_cast<int64_t>(std::is_integral<T>::value ? ((std::abs(limit - start)
                                                                   + std::abs(delta) - 1) / std::abs(delta))
                                                                 : std::ceil(std::abs((limit - start) / delta)));
  output->MutableTensorDesc().SetShape(GeShape());  // when size is 0

  if (size > 0) {
    unique_ptr<T[]> buf(new (std::nothrow) T[size]);
    if (buf == nullptr) {
      GELOGE(MEMALLOC_FAILED, "New buf failed.");
      return MEMALLOC_FAILED;
    }

    T val = start;
    for (int64_t i = 0; i < size; ++i) {
      buf[i] = val;
      val += delta;
    }
    if (output->SetData(reinterpret_cast<uint8_t *>(buf.get()), size * sizeof(T)) != GRAPH_SUCCESS) {
      GELOGW("GetRange: SetData failed");
    }
    output->MutableTensorDesc().SetShape(GeShape({size}));
  }

  return SUCCESS;
}

REGISTER_COMPUTE_NODE_KERNEL(RANGE, RangeKernel);
}  // namespace ge
