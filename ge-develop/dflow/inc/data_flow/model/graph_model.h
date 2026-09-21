/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_INC_MODEL_GRAPH_MODEL_H_
#define DFLOW_INC_MODEL_GRAPH_MODEL_H_
#include "dflow/inc/data_flow/model/pne_model.h"
namespace ge {
class GraphModel : public PneModel {
 public:
  explicit GraphModel(const ComputeGraphPtr compute_graph);
  ~GraphModel() override;

  Status Init(const ModelData &model_data);

  ModelData& GetModelData() {
    return model_data_;
  }

  Status SerializeModel(ModelBufferData &model_buffer) override;

  Status UnSerializeModel(const ModelBufferData &model_buffer) override;

  void SetModelId(uint32_t model_id) override {
    PneModel::SetModelId(model_id);
  }

 private:
  ModelData model_data_;
};

using GraphModelPtr = std::shared_ptr<GraphModel>;
}  // namespace ge
#endif  // DFLOW_INC_MODEL_GRAPH_MODEL_H_
