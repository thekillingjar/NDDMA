/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_PNE_DATA_FLOW_MODEL_PP_H_
#define INC_PNE_DATA_FLOW_MODEL_PP_H_

#include "inner_pp.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
namespace dflow {
class ModelPpImpl;
class GE_FUNC_VISIBILITY ModelPp : public InnerPp {
 public:
  ModelPp(const char_t *pp_name, const char_t *model_path);
  ~ModelPp() override = default;
  ModelPp &SetCompileConfig(const char_t *json_file_path);
 protected:
  void InnerSerialize(std::map<ge::AscendString, ge::AscendString> &serialize_map) const override;

 private:
  std::shared_ptr<ModelPpImpl> impl_;
};

}  // namespace dflow
}  // namespace ge
#endif  // INC_PNE_DATA_FLOW_MODEL_PP_H_
