/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pybind11/pybind11.h"
#include "graph/types.h"
#include "flow_func/tensor_data_type.h"

namespace {
namespace py = pybind11;
}

PYBIND11_MODULE(data_wrapper, m) {
  py::enum_<ge::DataType>(m, "DataType")
      .value("DT_FLOAT", ge::DataType::DT_FLOAT)
      .value("DT_FLOAT16", ge::DataType::DT_FLOAT16)
      .value("DT_BF16", ge::DataType::DT_BF16)
      .value("DT_INT8", ge::DataType::DT_INT8)
      .value("DT_INT16", ge::DataType::DT_INT16)
      .value("DT_UINT16", ge::DataType::DT_UINT16)
      .value("DT_UINT8", ge::DataType::DT_UINT8)
      .value("DT_INT32", ge::DataType::DT_INT32)
      .value("DT_INT64", ge::DataType::DT_INT64)
      .value("DT_UINT32", ge::DataType::DT_UINT32)
      .value("DT_UINT64", ge::DataType::DT_UINT64)
      .value("DT_BOOL", ge::DataType::DT_BOOL)
      .value("DT_DOUBLE", ge::DataType::DT_DOUBLE)
      .value("DT_STRING", ge::DataType::DT_STRING)
      .export_values();
  py::enum_<FlowFunc::TensorDataType>(m, "FuncDataType")
      .value("DT_FLOAT", FlowFunc::TensorDataType::DT_FLOAT)
      .value("DT_FLOAT16", FlowFunc::TensorDataType::DT_FLOAT16)
      .value("DT_BF16", FlowFunc::TensorDataType::DT_BF16)
      .value("DT_INT8", FlowFunc::TensorDataType::DT_INT8)
      .value("DT_INT16", FlowFunc::TensorDataType::DT_INT16)
      .value("DT_UINT16", FlowFunc::TensorDataType::DT_UINT16)
      .value("DT_UINT8", FlowFunc::TensorDataType::DT_UINT8)
      .value("DT_INT32", FlowFunc::TensorDataType::DT_INT32)
      .value("DT_INT64", FlowFunc::TensorDataType::DT_INT64)
      .value("DT_UINT32", FlowFunc::TensorDataType::DT_UINT32)
      .value("DT_UINT64", FlowFunc::TensorDataType::DT_UINT64)
      .value("DT_BOOL", FlowFunc::TensorDataType::DT_BOOL)
      .value("DT_DOUBLE", FlowFunc::TensorDataType::DT_DOUBLE)
      .export_values();
}