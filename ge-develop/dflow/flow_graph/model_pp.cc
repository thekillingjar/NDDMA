/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow/flow_graph/model_pp.h"
#include "common/util/mem_utils.h"
#include "framework/common/debug/ge_log.h"
#include "data_flow_attr_define.h"
#include "base/err_msg.h"

namespace ge {
namespace dflow {
class ModelPpImpl {
 public:
  explicit ModelPpImpl(const std::string &model_path) : model_path_(model_path) {}

  const std::string &GetModelPath() const {
    return model_path_;
  }

 private:
  std::string model_path_;
};

ModelPp::ModelPp(const char_t *pp_name, const char_t *model_path) : InnerPp(pp_name, INNER_PP_TYPE_MODEL_PP) {
  if (model_path == nullptr) {
    impl_ = nullptr;
    GELOGE(GRAPH_FAILED, "model path is nullptr.");
    return;
  }
  impl_ = MakeShared<ModelPpImpl>(model_path);
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "ModelPpImpl make shared failed.");
  }
}
void ModelPp::InnerSerialize(std::map<ge::AscendString, ge::AscendString> &serialize_map) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "InnerSerialize failed: ModelPpImpl can not be used, impl is nullptr.");
    return;
  }
  serialize_map[INNER_PP_CUSTOM_ATTR_MODEL_PP_MODEL_PATH] = impl_->GetModelPath().c_str();
}

ModelPp &ModelPp::SetCompileConfig(const char_t *json_file_path) {
  if (json_file_path == nullptr) {
    GELOGE(ge::FAILED, "[Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
           this->GetProcessPointName());
    return *this;
  }

  ProcessPoint::SetCompileConfigFile(json_file_path);
  GELOGI("SetCompileConfig, json_file_path=%s.", json_file_path);
  return *this;
}
}  // namespace dflow
}  // namespace ge
