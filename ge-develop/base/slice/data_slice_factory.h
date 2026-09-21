/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLICE_DATASLICE_DATA_SLICE_FACTORY_H_
#define SLICE_DATASLICE_DATA_SLICE_FACTORY_H_

#include <map>
#include <functional>
#include "graph/axis_type_info.h"
#include "ge/ge_api_types.h"
#include "slice/data_slice_infer_base.h"
namespace ge {
// automatic register factory
class DataSliceFactory {
 public:
  static DataSliceFactory *GetInstance() {
    static DataSliceFactory factory;
    return &factory;
  }
  std::shared_ptr<DataSliceInferBase> GetClassByAxisType(ge::AxisType axis_type);
  void RegistClass(ge::AxisType axis_type,
                     std::function<DataSliceInferBase*(void)> infer_util_class);
  ~DataSliceFactory() {}
 private:
  DataSliceFactory() {}
  std::map<ge::AxisType, std::function<DataSliceInferBase*(void)>> class_map_;
};

class AxisInferRegister {
 public:
  AxisInferRegister(ge::AxisType axis_type, std::function<DataSliceInferBase*(void)> data_slice_infer_impl) {
    DataSliceFactory::GetInstance()->RegistClass(axis_type, data_slice_infer_impl);
  }
  ~AxisInferRegister() = default;
};
}

#endif // SLICE_DATASLICE_DATA_SLICE_FACTORY_H_
