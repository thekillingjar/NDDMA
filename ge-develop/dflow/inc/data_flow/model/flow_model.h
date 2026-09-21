/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_MODEL_FLOW_MODEL_H_
#define GE_MODEL_FLOW_MODEL_H_

#include <map>
#include <mutex>
#include "graph/compute_graph.h"
#include "dflow/inc/data_flow/model/pne_model.h"

namespace ge {
class FlowModel : public PneModel {
 public:
  using PneModel::PneModel;
  explicit FlowModel(const ComputeGraphPtr &root_graph);
  ~FlowModel() override = default;

  Status SerializeModel(ModelBufferData &model_buff) override;

  Status UnSerializeModel(const ModelBufferData &model_buff) override;

  void SetModelsEschedPriority(std::map<std::string, std::map<std::string, int32_t>> models_esched_priority) {
    models_esched_priority_ = std::move(models_esched_priority);
  }

  const std::map<std::string, std::map<std::string, int32_t>> &GetModelsEschedPriority() const {
    return models_esched_priority_;
  }

  void SetLogicDeviceToMemCfg(std::map<std::string, std::pair<uint32_t, uint32_t>> logic_dev_id_to_mem_cfg) {
    logic_dev_id_to_mem_cfg_ = std::move(logic_dev_id_to_mem_cfg);
  }

  const std::map<std::string, std::pair<uint32_t, uint32_t>> &GetLogicDeviceToMemCfg() const {
    return logic_dev_id_to_mem_cfg_;
  }

 private:
  mutable std::mutex flow_model_mutex_;
  std::map<std::string, std::map<std::string, int32_t>> models_esched_priority_;
  // key logic_device_id | value(std_mem_size, shared_mem_size)
  std::map<std::string, std::pair<uint32_t, uint32_t>> logic_dev_id_to_mem_cfg_;
};
using FlowModelPtr = std::shared_ptr<ge::FlowModel>;
}  // namespace ge
#endif  // GE_MODEL_FLOW_MODEL_H_
