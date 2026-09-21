/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "graph/tensor.h"
#include "common/checker.h"
#include "ge_api_c_wrapper_utils.h"
#include "graph/utils/tensor_value_utils.h"
#include <iostream>
#include <memory>

#ifdef __cplusplus

using namespace ge;
using namespace ge::c_wrapper;
extern "C" {
#endif
struct EsCTensor;

graphStatus GeApiWrapper_Tensor_SetFormat(EsCTensor *tensor, const ge::Format &format) {
  GE_ASSERT_NOTNULL(tensor);
  auto *ts = static_cast<Tensor *>(static_cast<void *>(tensor));
  return ts->SetFormat(format);
}

ge::Format GeApiWrapper_Tensor_GetFormat(EsCTensor *tensor) {
  GE_ASSERT_NOTNULL(tensor);
  auto *ts = static_cast<Tensor *>(static_cast<void *>(tensor));
  return ts->GetFormat();
}

graphStatus GeApiWrapper_Tensor_SetDataType(EsCTensor *tensor, const ge::DataType &dtype) {
  GE_ASSERT_NOTNULL(tensor);
  auto *ts = static_cast<Tensor *>(static_cast<void *>(tensor));
  return ts->SetDataType(dtype);
}

ge::DataType GeApiWrapper_Tensor_GetDataType(EsCTensor *tensor) {
  GE_ASSERT_NOTNULL(tensor);
  auto *ts = static_cast<Tensor *>(static_cast<void *>(tensor));
  return ts->GetDataType();
}

EsCTensor *GeApiWrapper_Tensor_CreateTensor() {
  auto tensor = new (std::nothrow) ge::Tensor();
  GE_ASSERT_NOTNULL(tensor);
  return static_cast<EsCTensor *>(static_cast<void *>(tensor));
}

void GeApiWrapper_Tensor_DestroyEsCTensor(EsCTensor *tensor) {
  auto *ts = static_cast<Tensor *>(static_cast<void *>(tensor));
  delete ts;
}


graphStatus GeApiWrapper_Tensor_GetShape(EsCTensor *tensor, int64_t **dims, size_t *dims_num) {
  auto *ts = static_cast<Tensor *>(static_cast<void *>(tensor));
  auto dims_vec = ts->GetTensorDesc().GetShape().GetDims();
  *dims = VecDimsToArray(dims_vec, dims_num);
  return GRAPH_SUCCESS;
}

void GeApiWrapper_Tensor_FreeDimsArray(int64_t *dims) {
  delete[] dims;
}

const char *GeApiWrapper_Tensor_GetData(EsCTensor *tensor) {
  GE_ASSERT_NOTNULL(tensor);
  auto *ts = static_cast<Tensor *>(static_cast<void *>(tensor));
  auto data_type = ts->GetDataType();
  auto str_val = TensorValueUtils::ConvertTensorValue(*ts, data_type, ",", false);
  AscendString asc(str_val.c_str());
  return AscendStringToChar(asc);
}
#ifdef __cplusplus
}
#endif
