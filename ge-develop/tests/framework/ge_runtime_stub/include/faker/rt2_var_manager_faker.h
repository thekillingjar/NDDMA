/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_VAR_MANAGER_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_VAR_MANAGER_FAKER_H_
#include <string>
#include "base_node_exe_faker.h"
#include "stub/gert_runtime_stub.h"
#include "common/checker.h"
#include "framework/runtime/rt_var_manager.h"

namespace gert {
class UtestRt2VarManager : public RtVarManager {
 public:
  ge::Status GetVarShapeAndMemory(const std::string &id, gert::StorageShape &shape,
                                  gert::TensorData &memory) const override {
    GELOGI("Var manager %p Get variable %s", this, id.c_str());
    auto iter = named_variables_.find(id);
    GE_ASSERT(iter != named_variables_.end());

    shape = iter->second.first;
    memory.ShareFrom(iter->second.second);
    return ge::SUCCESS;
  }

  void SetVarShapeAndMemory(const std::string &id, gert::StorageShape &shape, gert::TensorData &memory) {
    GELOGI("Var manager %p Set variable %s", this, id.c_str());
    auto &shape_and_memory = named_variables_[id];
    shape_and_memory.first = shape;
    shape_and_memory.second.ShareFrom(memory);
  }

 private:
  std::map<std::string, std::pair<gert::StorageShape, gert::TensorData>> named_variables_;
};

class Rt2VarManagerFaker {
 public:
  Rt2VarManagerFaker(UtestRt2VarManager &variable_manager);
  ~Rt2VarManagerFaker() = default;
  void AddVar(const std::string &var_name, const std::vector<int64_t> &shape);
 private:
  UtestRt2VarManager *var_manager_;
  std::vector<std::unique_ptr<uint8_t[]>> value_holder_;
};
}
#endif // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_VAR_MANAGER_FAKER_H_
