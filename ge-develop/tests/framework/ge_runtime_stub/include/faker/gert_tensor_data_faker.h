/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_GERT_TENSOR_DATA_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_GERT_TENSOR_DATA_FAKER_H_
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/storage_format.h"
#include "multi_stream_allocator_faker.h"
namespace gert {
struct GtdHolder {
  StorageShape ss;
  StorageFormat sf;
  ge::DataType dt;
  TensorPlacement placement;
  int64_t stream_id{0};
  memory::MultiStreamAllocatorFaker::Holder allocator_holder;
  std::unique_ptr<GertTensorData> gtd;
};
class GertTensorDataFaker {
 public:
  GertTensorDataFaker &Shape(std::initializer_list<int64_t> shape);
  GertTensorDataFaker &OriginShape(std::initializer_list<int64_t> shape);
  GertTensorDataFaker &StorageShape(std::initializer_list<int64_t> shape);

  GertTensorDataFaker &Format(ge::Format format);
  GertTensorDataFaker &OriginFormat(ge::Format format);
  GertTensorDataFaker &StorageFormat(ge::Format format);
  GertTensorDataFaker &ExpandDimsType(gert::ExpandDimsType edt);

  GertTensorDataFaker &DataType(ge::DataType dt);

  GertTensorDataFaker &Placement(TensorPlacement placement);
  GertTensorDataFaker &StreamId(int64_t stream_id);

  GtdHolder Build() const;

 private:
  GtdHolder gtd_desc_;
};
}

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_GERT_TENSOR_DATA_FAKER_H_
