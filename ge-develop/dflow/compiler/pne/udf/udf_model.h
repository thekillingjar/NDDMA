/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_UDF_MODEL_H
#define UDF_UDF_MODEL_H

#include "dflow/inc/data_flow/model/pne_model.h"
#include "graph/compute_graph.h"
#include "proto/udf_def.pb.h"

namespace ge {
class UdfModel : public PneModel {
 public:
  UdfModel() = default;
  explicit UdfModel(const ComputeGraphPtr &root_graph);
  ~UdfModel() override = default;

  Status SerializeModel(ModelBufferData &model_buff) override;
  Status SerializeModelDef(ModelBufferData &model_buff) const;

  Status UnSerializeModel(const ModelBufferData &model_buff) override;

  udf::UdfModelDef &MutableUdfModelDef() {
    return udf_model_def_;
  }

  std::string GetLogicDeviceId() const override;

  std::string GetRedundantLogicDeviceId() const override;
 private:
  std::string GetLogicDeviceId(bool is_redundant) const;
  udf::UdfModelDef udf_model_def_;
};
using UdfModelPtr = std::shared_ptr<ge::UdfModel>;
}  // namespace ge
#endif  // UDF_UDF_MODEL_H
