/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_PNE_MODEL_SERIALIZED_MODEL_H_
#define BASE_PNE_MODEL_SERIALIZED_MODEL_H_

#include "dflow/inc/data_flow/model/pne_model.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
class SerializedModel : public PneModel {
 public:
  using PneModel::PneModel;
  ~SerializedModel() override = default;

  Status SerializeModel(ModelBufferData &model_buff) override {
    if (GetSavedModelPath().empty()) {
      model_buff.data = serialized_data_;
      model_buff.length = serialized_data_length_;
    } else {
      uint32_t data_len = 0U;
      std::string reosurce_file = GetSavedModelPath();
      auto buff = GetBinFromFile(reosurce_file, data_len);
      GE_CHECK_NOTNULL(buff, "open so fail, path=%s", reosurce_file.c_str());
      model_buff.data.reset(reinterpret_cast<uint8_t *>(buff.release()), std::default_delete<uint8_t[]>());
      model_buff.length = static_cast<uint64_t>(data_len);
    }
    return SUCCESS;
  }

  Status UnSerializeModel(const ModelBufferData &model_buff) override {
    serialized_data_ = model_buff.data;
    serialized_data_length_ = model_buff.length;
    return SUCCESS;
  }

  Status SetLogicDeviceId(const std::string &logic_device_id) override {
    logic_device_id_ = logic_device_id;
    return SUCCESS;
  }

  std::string GetLogicDeviceId() const override {
    return logic_device_id_;
  }

  Status SetRedundantLogicDeviceId(const std::string &logic_device_id) override {
    redundant_logic_device_id_ = logic_device_id;
    return SUCCESS;
  }

  std::string GetRedundantLogicDeviceId() const override {
    return redundant_logic_device_id_;
  }

 private:
  std::shared_ptr<uint8_t> serialized_data_ = nullptr;
  uint64_t serialized_data_length_ = 0;
  std::string logic_device_id_;
  std::string redundant_logic_device_id_;
};
}  // namespace ge
#endif  // BASE_PNE_MODEL_SERIALIZED_MODEL_H_
