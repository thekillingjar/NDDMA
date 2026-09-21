/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MODEL_DATA_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MODEL_DATA_FAKER_H_
#include <cstdint>
#include <memory>
#include "common/model/ge_root_model.h"
#include "common/ge_types.h"
namespace gert {
using EnvCreateFunc = void (*)(std::string opp_path, bool env_initialized);
class ModelDataFaker {
 public:
  class ModelDataHolder {
   public:
    ge::ModelData &Get() {
      return model_data_;
    }
   private:
    friend class ModelDataFaker;
    std::shared_ptr<uint8_t> data_holder_;
    ge::ModelData model_data_;
  };

 public:
  ModelDataFaker &GeRootModel(const ge::GeRootModelPtr &ge_root_model);
  ModelDataHolder BuildUnknownShape() const;
  void BuildUnknownShapeFile(const char *file_path) const;
  // 调用者需要保证ModelData 中model_data的正确释放
  ge::ModelData BuildUnknownShapeSoInOmFile(EnvCreateFunc env_create_func, std::string opp_path, bool env_initialized = false);

 private:
  ge::GeRootModelPtr ge_root_model_;
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MODEL_DATA_FAKER_H_
