/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/rt2_var_manager_faker.h"
#include "framework/common/taskdown_common.h"

namespace gert {
namespace {
Shape GeShapeToGertShape(const ge::GeShape &ge_shape) {
  Shape gert_shape;
  gert_shape.SetDimNum(ge_shape.GetDimNum());
  for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
    gert_shape.SetDim(i, ge_shape.GetDim(i));
  }
  return gert_shape;
}
}

Rt2VarManagerFaker::Rt2VarManagerFaker(UtestRt2VarManager &variable_manager) {
  var_manager_ = &variable_manager;
}
void Rt2VarManagerFaker::AddVar(const std::string &var_name, const std::vector<int64_t> &shape) {
  ge::GeTensorDesc var_desc;
  var_desc.SetShape(ge::GeShape(shape));
  var_desc.SetOriginShape(ge::GeShape(shape));
  var_desc.SetPlacement(ge::Placement::kPlacementDevice);

  gert::StorageShape gert_shape;
  gert_shape.MutableOriginShape() = GeShapeToGertShape(var_desc.GetOriginShape());
  gert_shape.MutableStorageShape() = GeShapeToGertShape(var_desc.GetShape());
  uint64_t size = sizeof(uint64_t);
  for (auto dim : shape) {
    size *= dim;
  }
  auto fake_value = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[size]);
  if (fake_value == nullptr) {
    return;
  }
  void *magic_addr = reinterpret_cast<void*>(fake_value.get());
  gert::TensorData gert_data(magic_addr);
  gert_data.SetSize(size);
  gert_data.SetPlacement(TensorPlacement::kOnDeviceHbm);
  var_manager_->SetVarShapeAndMemory(var_name, gert_shape, gert_data);
  value_holder_.emplace_back(std::move(fake_value.release()));
}
}
