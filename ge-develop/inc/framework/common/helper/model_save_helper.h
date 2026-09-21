/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_H_

#include <memory>
#include <string>

#include "framework/common/helper/model_save_helper_factory.h"
#include "framework/common/helper/om_file_helper.h"
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "framework/common/types.h"
#include "graph/model.h"
#include "platform/platform_info.h"
#include "common/op_so_store/op_so_store.h"

namespace ge {
class GE_FUNC_VISIBILITY ModelSaveHelper {
 public:
  ModelSaveHelper() = default;

  virtual ~ModelSaveHelper() = default;

  virtual Status SaveToOmRootModel(const GeRootModelPtr &ge_root_model,
                                   const std::string &output_file,
                                   ModelBufferData &model,
                                   const bool is_unknown_shape) = 0;

  virtual Status SaveToOmModel(const GeModelPtr &ge_model,
                               const std::string &output_file,
                               ModelBufferData &model,
                               const GeRootModelPtr &ge_root_model = nullptr) = 0;

  virtual void SetSaveMode(const bool val) = 0;

  // Configure from options - for attribute compression mode
  virtual Status ConfigureAttrCompressionMode(const string &mode) {
    (void)mode;
    return SUCCESS;
  }
 protected:
  ModelSaveHelper(const ModelSaveHelper &) = default;
  ModelSaveHelper &operator=(const ModelSaveHelper &) & = default;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_MODEL_SAVE_HELPER_H_
