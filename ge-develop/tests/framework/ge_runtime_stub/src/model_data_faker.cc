/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/model_data_faker.h"
#include "runtime/gert_api.h"
#include "common/helper/model_helper.h"
#include "common/helper/om_file_helper.h"
#include "graph/ge_local_context.h"
#include "mmpa/mmpa_api.h"
namespace gert {
ModelDataFaker::ModelDataHolder ModelDataFaker::BuildUnknownShape() const {
  if (ge_root_model_ == nullptr) {
    throw std::invalid_argument("The ge_root_model is nullptr");
  }
  ge::ModelBufferData model_buffer;
  ge::ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer
  if (model_helper.SaveToOmRootModel(ge_root_model_, "NoUse", model_buffer, true) != ge::SUCCESS) {
    throw std::invalid_argument("Failed to convert to ModelData");
  }
  ModelDataHolder holder;
  holder.data_holder_ = model_buffer.data;
  holder.model_data_.model_data = model_buffer.data.get();
  holder.model_data_.model_len = static_cast<uint32_t>(model_buffer.length);
  return holder;
}
ModelDataFaker &ModelDataFaker::GeRootModel(const ge::GeRootModelPtr &ge_root_model) {
  ge_root_model_ = ge_root_model;
  return *this;
}
void ModelDataFaker::BuildUnknownShapeFile(const char *file_path) const {
  if (ge_root_model_ == nullptr) {
    throw std::invalid_argument("The ge_root_model is nullptr");
  }
  ge::ModelBufferData model_buffer;
  ge::ModelHelper model_helper;
  model_helper.SetSaveMode(true);  // Save to file
  if (model_helper.SaveToOmRootModel(ge_root_model_, file_path, model_buffer, true) != ge::SUCCESS) {
    throw std::invalid_argument("Failed to convert to ModelData");
  }
}

ge::ModelData ModelDataFaker::BuildUnknownShapeSoInOmFile(EnvCreateFunc env_create_func, std::string opp_path, bool env_initialized) {
  if (ge_root_model_ == nullptr) {
    throw std::invalid_argument("The ge_root_model is nullptr");
  }
  env_create_func(opp_path, env_initialized);
  ge_root_model_->CheckAndSetNeedSoInOM();
  std::string output_file = "outputfile.om";
  ge::ModelBufferData model;
  ge::ModelHelper model_helper;
  if (model_helper.SaveToOmRootModel(ge_root_model_, output_file, model, true) != ge::SUCCESS) {
    throw std::invalid_argument("Failed to convert to ModelData");
  }

  std::string cpu;
  if (ge::GetThreadLocalContext().GetOption(ge::OPTION_HOST_ENV_CPU, cpu) != ge::GRAPH_SUCCESS) {
    throw std::invalid_argument("Failed to get env cpu option");
  }

  std::string os;
  if (ge::GetThreadLocalContext().GetOption(ge::OPTION_HOST_ENV_OS, os) != ge::GRAPH_SUCCESS) {
    throw std::invalid_argument("Failed to get env os option");
  }

  ge::ModelData model_data;
  std::string load_file = "outputfile_" + os + "_" + cpu + ".om";
  auto error_code = LoadDataFromFile(load_file.c_str(), model_data);

  system(("rm -rf " + load_file).c_str());
  if (error_code != ge::GRAPH_SUCCESS) {
    throw std::invalid_argument("Failed to load data from file");
  }
  // 调用者需要保证model_data.model_data正确释放
  return model_data;
}
}  // namespace gert
